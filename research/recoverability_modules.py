"""ROI memory, causal search utility, and relative action/risk learning.

Only DECISION_FIELDS enter forward. GT quality/future labels belong to objective.
The six search actions are the frozen visual regions executed by the collector.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .candidate_learning import CandidateQualityHead
from .temporal_modules import relative_geometry

DECISION_FIELDS = ('features', 'evidence', 'raw_score', 'boxes', 'instance_features',
                   'anchor_features', 'image_boxes', 'valid', 'c1_quality', 'original_choice',
                   'history_instance_descriptors', 'history_evidence', 'history_quality',
                   'history_boxes', 'history_frames', 'history_write', 'history_valid',
                   'motion_means', 'motion_log_weights')


class InstanceMemory(nn.Module):
    """Four slots per sensor: two target, one non-target, one uncertain."""
    def __init__(self, projection, hidden=128):
        super().__init__()
        self.projection = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, hidden), nn.GELU())
        self.projection.load_state_dict(projection.state_dict(), strict=True)
        self.gate = nn.Sequential(nn.Linear(hidden + 4, 64), nn.GELU(), nn.Linear(64, 3))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        with torch.no_grad():
            self.gate[-1].bias[0] = 1.  # Initial target probability .576 > verification threshold.
        self.target_keys = nn.Parameter(torch.randn(2, 2, hidden) * .02)

    def encode(self, features):
        return F.normalize(self.projection(features.float()), dim=-1)

    def initialize(self, anchor):
        return torch.cat((anchor[:, None].expand(-1, 2, -1, -1).clone(),
                          torch.zeros_like(anchor[:, None]).expand(-1, 2, -1, -1).clone()), 1)

    def gate_logits(self, descriptor, evidence, quality, anchor):
        point_anchor = evidence[..., [2, 4]]
        roi_anchor = (descriptor * anchor).sum(-1)
        cues = torch.stack((evidence[..., 0, None].expand_as(roi_anchor),
                            quality[..., None].expand_as(roi_anchor), point_anchor, roi_anchor), -1)
        return self.gate(torch.cat((descriptor, cues), -1))

    def write_rates(self, descriptor, probability, observed, commit):
        assignment = torch.einsum('bmd,smd->bsm', descriptor, self.target_keys).softmax(1)
        target_rate = assignment * probability[..., 0][:, None] * commit[:, None]
        other_rate = probability[..., 1:].permute(0, 2, 1) * observed[:, None, None]
        category = probability.argmax(-1)
        context_class = torch.stack(((category == 1) & (probability[..., 1] >= .5), category == 2), 1)
        other_rate *= context_class
        return torch.cat((target_rate, other_rate), 1)

    def update(self, memory, descriptor, probability, observed, commit):
        rates = self.write_rates(descriptor, probability, observed, commit)
        return memory * (1 - rates[..., None]) + descriptor[:, None] * rates[..., None]

    @staticmethod
    def support(memory, anchor, descriptor):
        target = F.normalize(torch.cat((anchor[:, None], memory[:, :2]), 1), dim=-1)
        positive = torch.einsum('bsmd,bkmd->bksm', target, descriptor).max(2).values
        other = F.normalize(memory[:, 2:], dim=-1)
        context = torch.einsum('bsmd,bkmd->bkms', other, descriptor)
        return torch.cat((positive[..., None], context), -1)

    def history(self, data):
        anchor = self.encode(data['anchor_features'])
        past = self.encode(data['history_instance_descriptors'])
        memory = self.initialize(anchor)
        logits = []
        for frame in range(past.shape[1]):
            gate = self.gate_logits(past[:, frame], data['history_evidence'][:, frame].float(),
                                    data['history_quality'][:, frame].float(), anchor)
            observed = data['history_valid'][:, frame] & (data['history_frames'][:, frame] > 0)
            probability = gate.softmax(-1)
            verified = probability[..., 0] >= .5
            commit = observed[:, None] & data['history_write'][:, frame, None] & verified
            memory = self.update(memory, past[:, frame], probability, observed, commit)
            logits.append(gate)
        return anchor, memory, past, torch.stack(logits, 1)


class RecoverabilityModules(nn.Module):
    def __init__(self, c1_checkpoint, hidden=128):
        super().__init__()
        assert hidden == c1_checkpoint['args']['hidden']
        self.c1 = CandidateQualityHead(hidden)
        self.c1.load_state_dict(c1_checkpoint['head'], strict=True)
        self.c1.eval().requires_grad_(False)
        self.memory = InstanceMemory(self.c1.projection, hidden)
        self.action = nn.Sequential(nn.Linear(hidden * 5 + 30, hidden), nn.GELU(), nn.Linear(hidden, 7))
        # Two local summaries plus causal motion geometry/distribution.
        self.search = nn.Sequential(nn.Linear((hidden * 3 + 23) * 2 + 39 + 48, hidden),
                                    nn.GELU(), nn.Linear(hidden, 13))
        for module in (self.action, self.search):
            nn.init.zeros_(module[-1].weight)
            nn.init.zeros_(module[-1].bias)
        with torch.no_grad():
            self.action[-1].bias[3:6] = -3.
            self.search[-1].bias[-1] = -3.

    def train(self, mode=True):
        super().train(mode)
        self.c1.eval()
        return self

    def candidate_inputs(self, data, anchor, memory):
        batch = len(data['valid'])
        descriptor = self.memory.encode(data['instance_features']).reshape(batch, 35, 2, -1)
        evidence = data['evidence'].float().reshape(batch, 35, 10)
        quality = data['c1_quality'].float().reshape(batch, 35)
        support = self.memory.support(memory, anchor, descriptor)
        gate = self.memory.gate_logits(descriptor, evidence, quality, anchor[:, None])
        proof = gate.softmax(-1)[..., 0]
        with torch.no_grad():
            fused = self.c1.projection(data['features'].float()).reshape(batch, 35, -1)
        reference = data['history_boxes'][:, -1].float()
        coordinates = relative_geometry(data['image_boxes'].float().reshape(batch, 35, 4), reference)
        # Per-candidate causal evidence; search consumes region0 only.
        inputs = torch.cat((fused, descriptor.flatten(-2), evidence, support.flatten(-2),
                            proof, coordinates, quality[..., None]), -1)
        return inputs, fused, descriptor, support, gate, proof

    def search_decision(self, inputs, data):
        local = inputs[:, :5]
        valid = data['valid'][:, 0]
        mean = (local * valid[..., None]).sum(1) / valid.sum(1, keepdim=True)
        maximum = local.masked_fill(~valid[..., None], -torch.inf).max(1).values
        boxes = data['history_boxes'][:, -8:].float()
        coordinates = relative_geometry(boxes, data['history_boxes'][:, -1].float())
        time = (data['history_frames'][:, -8:] - data['history_frames'][:, -1:]).float() / 8
        past_valid = data['history_valid'][:, -8:]
        motion_history = torch.cat((coordinates, time[..., None], past_valid.float()[..., None]), -1)
        motion_history *= past_valid[..., None]
        query = torch.cat((mean, maximum, data['motion_means'].float().flatten(1),
                           data['motion_log_weights'].float(), motion_history.flatten(1)), -1)
        result = self.search(query)
        return {'region_advantage': result[:, :6].tanh(),
                'region_success_logits': result[:, 6:12], 'absence_logit': result[:, 12]}

    def forward(self, data):
        anchor, memory, past, past_logits = self.memory.history(data)
        output = self.decide(data, anchor, memory)
        output.update(past_descriptors=past, past_gate_logits=past_logits)
        return output

    def decide(self, data, anchor, memory):
        """Current-only decision from the incrementally maintained bounded state."""
        inputs, fused, descriptor, support, gate, proof = self.candidate_inputs(data, anchor, memory)
        batch = len(inputs)
        keep = data['original_choice'].long()
        keep_fused = fused[torch.arange(batch, device=inputs.device), keep]
        keep_roi = descriptor[torch.arange(batch, device=inputs.device), keep].mean(1)
        region = torch.eye(7, device=inputs.device)[None, :, None].expand(batch, -1, 5, -1).reshape(batch, 35, 7)
        action_inputs = torch.cat((inputs, keep_fused[:, None].expand(-1, 35, -1),
                                   keep_roi[:, None].expand(-1, 35, -1), region), -1)
        result = self.action(action_inputs).reshape(batch, 7, 5, 7)
        raw_advantage = result[..., 1:3].tanh()
        reference = raw_advantage[torch.arange(batch, device=inputs.device), 0, keep, 0]
        harm = result[..., 3:5].sigmoid()
        baseline_harm = harm[torch.arange(batch, device=inputs.device), 0, keep, 0]
        scores = raw_advantage - reference[:, None, None, None]
        scores -= .1 * (harm - baseline_harm[:, None, None, None]).clamp(min=0)
        query_risk = result[..., 5].sigmoid() * (data['raw_score'] > .84)
        keep_risk = query_risk[torch.arange(batch, device=inputs.device), 0, keep]
        action_risk = torch.stack((query_risk, torch.zeros_like(query_risk)), -1)
        scores -= .025 * (action_risk - keep_risk[:, None, None, None])
        scores[:, 1:] -= .01  # Cost of one extra visual search, not six.
        output = {'scores': scores, 'advantage': raw_advantage - reference[:, None, None, None],
                  'harm_logits': result[..., 3:5], 'write_risk_logits': result[..., 5],
                  'quality_logits': torch.logit(data['c1_quality'].float().clamp(1e-5, 1-1e-5)) + result[..., 0],
                  'future_quality': result[..., 6].sigmoid(),
                  'target_probability': proof.reshape(batch, 7, 5, 2),
                  'gate_logits': gate.reshape(batch, 7, 5, 2, 3),
                  'descriptors': descriptor, 'support': support, 'anchor': anchor,
                  'memory': memory}
        output.update(self.search_decision(inputs, data))
        return output


def select_actions(output, data, threshold=.03, write_verification='identity', search_value='weighted'):
    """One optional extra region, then change only with positive net advantage."""
    assert write_verification in ('identity', 'action')
    assert search_value in ('weighted', 'gross')
    batch = len(data['valid'])
    device = data['valid'].device
    keep = data['original_choice'].long()
    regional = output['region_advantage']
    if search_value == 'weighted':
        regional = regional * output['region_success_logits'].sigmoid()
    # The gross target already measures utility gain, including partial recovery.
    regional = regional - .01
    best_region = regional.argmax(1) + 1
    current_quality = output['quality_logits'].sigmoid()[torch.arange(batch, device=device), 0, keep]
    search = ((output['absence_logit'].sigmoid() >= .5) | (current_quality < .5)) & (regional.max(1).values > threshold)
    valid = data['valid'][..., None].expand(-1, -1, -1, 2).clone()
    writes = data['raw_score'] > .84
    valid[..., 1] &= writes
    write_allowed = (output['target_probability'] >= .5).all(-1)
    if write_verification == 'action':
        write_allowed = torch.ones_like(write_allowed)
    valid[..., 0] &= ~writes | write_allowed
    available = torch.zeros((batch, 7), dtype=torch.bool, device=device)
    available[:, 0] = True
    available[torch.arange(batch, device=device), best_region] = search
    valid &= available[:, :, None, None]
    keep_pause = (writes[torch.arange(batch, device=device), 0, keep]
                  & ~write_allowed[torch.arange(batch, device=device), 0, keep]).long()
    keep_index = keep * 2 + keep_pause
    scores = output['scores'].masked_fill(~valid, -torch.inf).flatten(1)
    baseline = scores.gather(1, keep_index[:, None]).squeeze(1)
    best_value, best_index = scores.max(1)
    selected = torch.where(best_value > baseline + threshold, best_index, keep_index)
    return {'flat_action': selected, 'region': selected // 10, 'candidate': selected % 10 // 2,
            'pause': selected % 2, 'search_triggered': search, 'searched_region': best_region,
            'net_advantage': scores.gather(1, selected[:, None]).squeeze(1) - baseline}


def action_utility(data):
    return (.7 * data['current_iou'].float()[..., None]
            + .3 * data['future_iou'].float().mean(-1) - .1 * data['wrong_update_fraction'].float())


@torch.no_grad()
def search_supervision_targets(output, data, mode='oracle', threshold=.03, write_verification='identity'):
    """Gross search gain; deployment charges .01 once per extra visual forward.

    Selector targets compare the same detached selector with and without each
    extra region. GT is looked up after its choices; it never selects an action.
    The continuation utility still comes from the frozen-C1 cache teacher.
    """
    utility = action_utility(data)
    rows = torch.arange(len(data['valid']), device=utility.device)
    current = data['current_iou'].float()
    if mode == 'oracle':
        reference = utility[rows, 0, data['original_choice'].long(), 0]
        region_best = utility.masked_fill(~data['action_valid'], -torch.inf).flatten(2).max(-1).values
        has_candidate = data['action_valid'].flatten(2).any(-1)
        region_best = torch.where(has_candidate, region_best, reference[:, None])
        gain = region_best[:, 1:] - reference[:, None]
        success = (current.masked_fill(~data['valid'], -1).max(-1).values[:, 1:] >= .5).float()
        return gain, success
    assert mode == 'selector'
    forced = dict(output)
    forced['region_advantage'] = torch.full_like(output['region_advantage'], -1)
    forced['region_success_logits'] = torch.zeros_like(output['region_success_logits'])
    forced['absence_logit'] = torch.full_like(output['absence_logit'], 20)
    local = select_actions(forced, data, threshold, write_verification)
    assert not local['search_triggered'].any() and (local['region'] == 0).all()
    reference = utility.flatten(1)[rows, local['flat_action']]
    gains, successes = [], []
    for region in range(1, 7):
        forced['region_advantage'] = torch.full_like(output['region_advantage'], -1)
        forced['region_advantage'][:, region - 1] = 1
        chosen = select_actions(forced, data, threshold, write_verification)
        assert chosen['search_triggered'].all() and (chosen['searched_region'] == region).all()
        assert ((chosen['region'] == 0) | (chosen['region'] == region)).all()
        selected = utility.flatten(1)[rows, chosen['flat_action']]
        has_candidate = data['valid'][:, region].any(1)
        gains.append(torch.where(has_candidate, selected - reference, 0))
        selected_current = current[rows, chosen['region'], chosen['candidate']]
        successes.append(((selected_current >= .5) & has_candidate).float())
    return torch.stack(gains, 1), torch.stack(successes, 1)


def budgeted_winner_loss(scores, utility, action_valid, reference, keep, threshold):
    """Teach the deployed winner against its current rival within each search budget."""
    batch, region_count, candidates, actions = scores.shape
    scores, values, legal = scores.flatten(1), utility.flatten(1), action_valid.flatten(1)
    rows = torch.arange(batch, device=scores.device)
    indices = torch.arange(scores.shape[1], device=scores.device)
    regions = indices // (candidates * actions)
    nonkeep = indices[None] != (keep * actions)[:, None]
    decision_scores = scores - threshold * nonkeep
    decision_values = (values - reference[:, None] - .01 * (regions > 0)[None]
                       - threshold * nonkeep)
    losses = []
    for region in range(region_count):
        available = legal & ((regions == 0) | (regions == region))[None]
        best_value, winner = decision_values.masked_fill(~available, -torch.inf).max(1)
        winner = torch.where(best_value > 0, winner, keep * actions)
        rivals = available & (indices[None] != winner[:, None])
        rival_score, rival = decision_scores.masked_fill(~rivals, -torch.inf).max(1)
        regret = decision_values[rows, winner] - decision_values[rows, rival]
        losses.append(F.relu(regret + rival_score - decision_scores[rows, winner]))
    return torch.stack(losses, 1).mean()


def ranking_supervision_loss(scores, utility, action_valid, reference, mode='reference'):
    if mode == 'reference':
        delta = utility - reference[:, None, None, None]
        meaningful = action_valid & (delta.abs() > .05)
        return (F.relu(.03 - delta.sign() * scores) * meaningful).sum() / meaningful.sum().clamp(min=1)
    assert mode == 'pairwise'
    # Competitors must coexist within original plus ONE extra visual region.
    scores = scores.flatten(1)
    values, legal = utility.flatten(1), action_valid.flatten(1)
    regions = torch.arange(scores.shape[1], device=scores.device) // 10
    compatible = ((regions[:, None] == 0) | (regions[None, :] == 0)
                  | (regions[:, None] == regions[None, :]))
    gap = values[:, :, None] - values[:, None, :]
    pairs = legal[:, :, None] & legal[:, None, :] & compatible & (gap > .05)
    weight = gap.clamp(min=0) * pairs
    predicted_gap = scores[:, :, None] - scores[:, None, :]
    return (F.relu(.03 - predicted_gap) * weight).sum() / weight.sum().clamp(min=1)


def write_pair_supervision_loss(scores, utility, action_valid):
    """Calibrate net pause-versus-write scores only where both actions exist."""
    predicted = scores[..., 1] - scores[..., 0]
    target = utility[..., 1] - utility[..., 0]
    paired = action_valid[..., 1]
    error = F.smooth_l1_loss(predicted, target, reduction='none', beta=.01)
    return (error * paired).sum() / paired.sum().clamp(min=1)


def objective(model, output, data, search_supervision='oracle', threshold=.03, action_ranking='reference',
              write_pair_calibration=False, write_verification='identity'):
    batch = len(data['valid'])
    valid, action_valid = data['valid'], data['action_valid']
    current = data['current_iou'].float()
    utility = action_utility(data)
    rows = torch.arange(batch, device=utility.device)
    keep = data['original_choice'].long()
    reference = utility[rows, 0, keep, 0]
    delta = utility - reference[:, None, None, None]
    harmful = (delta < -.05) | ((current[rows, 0, keep, None, None, None] >= .5) & (current[..., None] < .2))
    advantage_loss = F.smooth_l1_loss(output['advantage'][action_valid], delta[action_valid])
    harm_loss = F.binary_cross_entropy_with_logits(output['harm_logits'][action_valid], harmful.float()[action_valid])
    if action_ranking == 'budgeted':
        ranking_loss = budgeted_winner_loss(output['scores'], utility, action_valid, reference,
                                           keep, threshold)
    else:
        ranking_loss = ranking_supervision_loss(output['scores'], utility, action_valid, reference, action_ranking)
    quality_loss = F.binary_cross_entropy_with_logits(output['quality_logits'][valid], current[valid])
    write_risk = ((data['raw_score'] > .84) & (current < .2)).float()
    risk_loss = F.binary_cross_entropy_with_logits(output['write_risk_logits'][valid], write_risk[valid])
    future_loss = F.mse_loss(output['future_quality'][valid], data['future_iou'].float().mean(-1)[..., 0][valid])

    # No overlap is a spatial non-target label, not a distractor semantic identity.
    def categories(quality):
        return torch.where(quality >= .5, 0, torch.where(quality == 0, 1, 2)).long()
    candidate_category = categories(current)[..., None].expand(-1, -1, -1, 2)
    gate_loss = F.cross_entropy(output['gate_logits'][valid].reshape(-1, 3), candidate_category[valid].reshape(-1))
    known_past = data['history_valid'] & (data['history_iou'] >= 0) & (data['history_frames'] > 0)
    history_category = categories(data['history_iou'])[..., None].expand(-1, -1, 2)
    past_ce = F.cross_entropy(output['past_gate_logits'].flatten(0, 2), history_category.flatten(), reduction='none').reshape(batch, -1, 2)
    gate_loss += .2 * (past_ce * known_past[..., None]).sum() / (known_past.sum() * 2).clamp(min=1)

    trusted = data['history_valid'] & (data['history_iou'] >= .5)
    sources = torch.cat((output['anchor'][:, None].detach(), output['past_descriptors'].detach()), 1)
    source_valid = torch.cat((torch.ones_like(trusted[:, :1]), trusted), 1)
    teacher = torch.einsum('btmd,bkmd->bktm', sources, output['descriptors'].detach())
    teacher = teacher.masked_fill(~source_valid[:, None, :, None], -torch.inf).max(2).values.mean(-1)
    student = output['support'][..., 0].mean(-1)
    flat_current, flat_valid = current.flatten(1), valid.flatten(1)
    pairs = flat_valid[:, :, None] & flat_valid[:, None, :] & (flat_current[:, :, None] >= .5) & (flat_current[:, None, :] == 0)
    teacher_gap = teacher[:, :, None] - teacher[:, None, :]
    student_gap = student[:, :, None] - student[:, None, :]
    preservation = ((student_gap - teacher_gap).square() * pairs).sum() / pairs.sum().clamp(min=1)
    identity = (F.relu(.2 - student_gap) * pairs).sum() / pairs.sum().clamp(min=1)

    region_gain, region_success = search_supervision_targets(output, data, search_supervision, threshold,
                                                           write_verification)
    region_value_loss = F.smooth_l1_loss(output['region_advantage'], region_gain)
    search_success_loss = F.binary_cross_entropy_with_logits(output['region_success_logits'], region_success)
    absence = (current[:, 0].masked_fill(~valid[:, 0], -1).max(-1).values < .5).float()
    absence_loss = F.binary_cross_entropy_with_logits(output['absence_logit'], absence)
    beneficial_search = region_gain.max(-1).values > .05
    search_scores = output['region_advantage'] * output['region_success_logits'].sigmoid() - .01
    search_ce = F.cross_entropy(search_scores / .1, region_gain.argmax(1), reduction='none')
    search_rank = (search_ce * beneficial_search).sum() / beneficial_search.sum().clamp(min=1)
    loss = (advantage_loss + ranking_loss + .2 * harm_loss + .2 * quality_loss + .1 * risk_loss
            + .1 * future_loss + .3 * gate_loss + .1 * (preservation + identity)
            + region_value_loss + .2 * (search_success_loss + absence_loss) + .2 * search_rank)
    parts = {key: float(value.detach()) for key, value in {
        'advantage': advantage_loss, 'action_ranking': ranking_loss, 'harm': harm_loss,
        'quality': quality_loss, 'query_write_risk': risk_loss, 'future': future_loss,
        'memory_gate': gate_loss, 'margin_preservation': preservation, 'identity_margin': identity,
        'search_value': region_value_loss, 'search_success': search_success_loss,
        'candidate_absence': absence_loss, 'search_ranking': search_rank}.items()}
    if write_pair_calibration:
        write_pair_loss = write_pair_supervision_loss(output['scores'], utility, action_valid)
        loss = loss + write_pair_loss
        parts['query_write_advantage'] = float(write_pair_loss.detach())
    return loss, parts
