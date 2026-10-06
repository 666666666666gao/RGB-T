# 实验跟踪（2026-10-02）

| 任务 | 状态 | 证据/备注 |
|---|---|---|
| 官方源码与完整权重 | DONE | upstream commit 339c737cda6a24be667b6e5abdc721e8d6046f05；gola_b224.bin 与 DINOv2 主干缓存 |
| 环境 GPU smoke | PASS | /data/gb/setup/smoke_gola.json；随机权重，仅环境验收 |
| 官方作者结果校准 | DONE | LasHeR 77.5/73.9/61.6，RGBT234 92.2/69.5；作者预测/包GT，不是新推理，RGBT234包GT与实际数据3文件不同 |
| 原 pretrained eval | FAILED | 18:47:38 Gloo 默认 30 分钟超时，未进入推理 |
| 原基线 10 epoch 训练 | STOPPED | 20:31:54 按用户新优先级停止；20:32:37 确认无进程，尚未 GPU minibatch |
| LasHeR train 索引 | DONE | 979/979 完成；MMOT .np 与 Video .pkl 保留 |
| C1 代码审查 | PASS | same-family provisional；无BLOCKING；val默认值已与计划对齐 |
| C1-sanity | PASS | 20:59:16启动；8steps×4；新参数117889更新、base无梯度、有限loss；8验证样本无失败，不证明恢复 |
| C1-initial | DONE | 21:01启动，21:08左右完成；12288训练采样；256val IoU .744937→.750184，4/8重新选对，2/227正确样本变错 |
| C2/C3/A/B | NOT IMPLEMENTED | 固定分支、状态恢复、rollout、判别压缩、多未来尚待实现 |
| 官方测试与主结果 | PARTIAL | RGBT234234seq/116649frames actualGT DONE；C1整体精度基本持平；LasHeR test在运行；完整方法未实现；VTUAV deferred |
| GitHub上传 | DONE INITIAL PUSH | 补全upstream历史后d6f1544成功push；后续完整结果/文档待追加提交 |
| Online代码审查 | PASS | independent same-family provisional；修正训练/推理dtype、attention、计时记录 |
| Online C1/baseline smoke | PASS | 四个128帧检查；首框/数量/有限值/延迟核验PASS，截断仅功能检查 |
| Online核心数据集完整评测 | RGBT234 DONE / LasHeR RUNNING | 22:35 LasHeR两原session活跃90/245、73440帧，ETA约105分钟；RGBT234两组234/234、116649帧全量完成，actualGT来源已验证 |
| RGBT234真实GT来源修正 | PASS / DEPLOYED | fresh reviewPASS；原生loader逐序列/模态及图像数PASS；只CPU重评分、保留旧包GT日志、无推理重跑 |
| RGBT234正式C1初轮效果 | INCONCLUSIVE / ESSENTIALLY TIED | baseline MPR91.778858/MSR69.232593；C1 MPR91.817009/MSR69.233890，差+.038151/+.001297百分点；继续C2/C3前须分析收益/损害，不主张显著提升 |

## 2026-10-03 integrated ABC campaign (current)
| Stage | Status | Evidence / next gate |
|---|---|---|
| Original fullGOLA-B/C1 two-dataset evaluation | COMPLETE | e72b9bb full native/all diagnostics/5CI include0 |
| New ABC acceptance | ACTIVE_UNMET | all5 overall standard metrics >=baseline+2pp, no narrowed claim |
| A/B/C + causal collector/train/online interfaces | SOURCE_PASS | abc_temporal_review final PASS for M0; whole-prefix256 causal training |
| Complete-prefix memory training correction | IMPLEMENTED_PREDEPLOY | reviewer actual24frame mismatch corrected to256>=maxprefix |
| Real new ABC sanity | PASS_NO_GAIN_CLAIM | actual8TRAIN/8val;2optimizersteps;ABCchanged/C1frozen;2x128 onlineframes;0switches |
| Collection / joint training | M1_RUNNING_M2_PENDING | three GPUs1/2/3 train704/640/416of1024 at10:14;val512 COMPLETE;fixed30epoch/batch64 chain WAIT_COLLECTION;GPU0otheraccount occupied |
| Full ABC official benchmark | NOT_RUN | after internalvalidation, all245/234 andcomplete native/branch-aware metrics |

| New full ABC diagnostics/audit | SOURCE_AND_GPU_M0_PASS | actualtemplate/learnedslot provenance,forecastPIT/NLL,strict8valreload;0real GPU switches, fullbench pending |

### 2026-10-03 11:28 actual M1/M2 completion and predeclared M3 selection
All three1024TRAIN collectors and512validation complete; 3072sampled/3032unique/40repeat TRAIN queries. Joint30epochs/1440steps/B64/seed42: A109569/B48843/C18691 changed, frozenC1exact/no gradients;707.236813s/9638.287109MiB. All31cached selectedIoU tiesC1 .749872207642;best0,0reselect/rescue/harm. Initial/best/last strictall512reloaddiff0 and all31epochs retained; calibration improved withoutchoicegain. Full98online TRAINvalidation49418frames each for initial/last/last_box_only running GPUs2/0/3; predeclare sequence-weighted valid-frameIoU for initial-vs-last selection BEFORE official test. Box-only is state-repair control, not checkpoint candidate. None of these are official PR/NPR/SR, no +2 claim; fullcore245/234 all31attrs/native/branch/forecast/cost/CI stillrequired. A new offline98collector is under limited same-family review; original algorithm frozen.
### 11:40 all98videos COMPLETE / frozenlast30 / full4GPU officialRUNNING
Allinitial/last/lastbox98seq49418frames49233valid complete. Predeclaredsequenceweighted fullonlineIoU initial .737194672342,last .739523904518,box .738778577820; selectlast30 at11:39:17 BEFORE officialABC test, tinygain+.232923pp vsinit NOTbaseline/+2. Detailed branch/slotforecast CPUcollector continues independently, compareaccuracywithselectionreceiptwhencomplete; do not waitCPUmechanisms to keepGPUuseful. run_abc_full.sh continuedreviewPASS noalgchange/fulloldbaseline receiptsverified. Fourofficial jobs launched11:40:05 GPUs0lasabc/1rgbabc/2lasbox/3rgbbox; full245/22070319attrs and234/11664912attrs/native5metrics/allseq/curves/events/branchslots/forecast/cost/5000pairedCI, fixedepoch30/P3W5patience2/seed42. RUN dirs /data/gb/outputs/abc_full_v1/{lasher_abc,rgbt234_abc,lasher_abc_box_only,rgbt234_abc_box_only}. FinalmarkeronlyallstagesPASS; sourcefrozen whilelive. NoformalABCscoreyet/all5+2UNMET. Fulltraining31epochPNG/PDFretained.
### 11:49 all98 mechanism report completed / negative uncertainty retained
CPUfull internal report finished; all3accuracyexactlockedselection. Initial/last/box failedframes8018/7487/7477,failureevents235/217/216,recovery222/235,206/217,205/216; switches514/446/463,benefit79/85/84,harm53/56/62. last-init+.232923ppCI[-.345903,.896431],23better13worse62ties. last-box+.074533ppCI[-.210859,.423822],8better10worse80ties. AllCIinclude0, fullstateNOTallmetricsbetter (RGB/TIRslotusedwrongfractions higherthanbox;217vs216events). Realtemplate/slot/write/forecast/calibration/provenance/all294sequence rows retained, nosemanticIDclaims. Fourfullofficialjobscontinue fixedepoch30; all5+2 UNMET. Add fullnativeABC-vs-boxmatchedreport afterbothfullsuccessmarkers, noextraGPU/inference/params.
### 2026-10-03 12:25 TRAIN-only search audit COMPLETE_NO_GAIN_CLAIM
Actual CPUreadonly full3072TRAIN/512val fixedepoch30 auditv2 nativefactor4/min10 geometry. TRAIN189fails/151candidate-missing,88missingoutside-original,0topmode/21any3modeextra-centercoverage; val33fails/25missing,11outside,0top/4any. Fourregions sum4xarea does NOTprovevisualrecall/accuracy; oldinlinegeometryobsoletepreserved. Noofficialtestparameters/checkpoint/sourcealgorithmchanged; new177103 A/B/Cparams counted; continuedlimited review/CPU8witness passed. FullfourGPU official and2CPUstatecompare unchanged, nextmeaningfulcheck12:45. Allfive+2 UNMET, no newtracker launched.
## 2026-10-03 12:48 complete-core CPU merge -- continued limited PASS and actual WAIT launch
Existing abc_metrics_review gpt-6-astra/max same-family/provisional, NOTfreshcontext. Actual independent full-schema fixture called deployed main:1437seqrows/93attrs/5acceptance; six5000bootstrapCIs;4pass1fail/negative-state preserved. Missingstate marker/missingseqrow/CImean mismatch each rejectedbeforeoutput. Fulltrainingreload/search audit,31epochs,oldbaseline/C1mechanisms andABCdiagnostics preserved exactly; currentparams88388437. compile/bash-n andlocal/servercodeequal PASS. Evidence abc_complete_merge_review.json, notactualtrackingresults. Finalizeractuallylaunched abc_full_complete_merge_20261003 afterPASS; waits240s forfourfull+twostate markers, singleCPUmerge/noGPU/retry. Official12:45 all4live/nonecomplete:Las80seq65842each;RGBmain196seq99523,box203seq103597. NextmeaningfulRGBcheck13:15, no partialtesttuning. Allfive+2UNMET.

## 2026-10-06T07:07:09.435730+08:00 selective-state-commit execution update

Actual four-GPU query M0 PASS, full1789labels collecting original3338360. Original3351735 queued: four2epoch fit sanities, fourfull60epoch C-extension fits (H3/H32 × lost0/.1, B128 LR.001 seed42), all98video selection, onehead forboth245/234 native/full5/31attrs/4pairedreferences. A/B+completeGOLA/C1/old4 remain active/frozen in this prototype; not jointABC retraining. Actual consecutive known-state events get totaltrainingweight1. Two27/32frame videos explicitlyH3-only; H32 doesnot inventfuture labels. OptimizerVAL exclusion, nofutureGTinput, no TEST checkpointchoice; onebestmodel sufficient, allfiveGOLA+2 stillUNMET. Humanhandoff only docs/HANDOFF_20261002.md; authoritative machine snapshot refine-logs/runs/selective_state_commit/current_execution_snapshot.json.


## 2026-10-06T07:45:41.497027+08:00 pre-fit endpoint-selection correction

Actual waiting-only3351735 retired with no fit/children/updates. Replacement3423188 waits unchanged3338360 collection; no NN collection/training replay. Four full60 fits preserve cached-best and actual60endpoint, strictreloadboth. All eight candidates each full98/49418 in two4GPU waves; selectONE by developer sequenceIoU before samehead bothnative/all5/31attrs/4refs5000CIs. Retire11 own unused heads onlyafter all consumers. Recipe/data/model unchanged, no>=3positive-seed gate, targetall5GOLA+2 remainsUNMET.


## 2026-10-06T07:57:55.811531+08:00 CPU mechanism-report coverage

New selective fields aggregate actual same-frame GT search/motion geometry localization, mask unknownGT, couple motion failure to correct active appearance-template source, verify actual mode2 count. SourcePASS samefamily/provisional; four-state mask fixture and actual333frame legacy counters/bins/interventions PASS. Synthetic extension is not selective NN runtime or causal benefit. NN/train/eval replay0; current collectors/fit/native queue unchanged, all5+2UNMET.


## 2026-10-06T10:20:55.846949+08:00 actual full-training/internal closure, native ongoing

Four C-extension fits each60/840 (not jointbackbone training), eightfull98/49418 complete; all learned checkpoints below frozenold4/gross. One epoch0 locked before bothnative; actual98prediction text equalparent, no training gain. TRAIN1646event1691queries/881seq and615write-pairqueries; threeeventroles overlap. Actual16CPUcachedforwards show TRAINpositive/VALnegative selectedutility despitepositive predictedutility. History244CSV/plots and report metadata copied; no NN/report replay. Currentoriginal3423188/native3643976-79 lastactual09:59:49, sole22853 next10:53:18 then240. Five formal new metrics pending; all5+2UNMET. One best sufficient, old4/dependencies protected.


## 2026-10-06T10:28:16.939834+08:00 matched write/motion commit CPU contrasts

H32 lost0 pause-vs-regular613TRAIN/55VAL queries; lowraw mode2-vs-samecandidate-mode0 1208TRAINpairs/1009events and58VALpairs/56events. Search reference plus motionhistory box+quality copying, not coordinate-only; GTunknown masked/eventequal. Positive contrast means reduced alternate-action harm, not better than parentkeep or deployed C efficacy. Reviewed explanation corrected; all16numericrows unchanged, no NN/optimizer/replay. Formal5pending; +2ACTIVE_UNMET.


## 2026-10-06T11:09:59.304261+08:00 actual full native closure; goal still unmet

Four60/840 fits, eightfull98 and SAME epoch0 both245/220703+234/116649 actual-GT complete. Five78.016094/74.383784/62.072660 and92.866361/70.022381, allbaseline+2false; exactgrossparent scores/no learned C benefit. Las threeCIpositive/RGBtwoinclude0; all31attrs/81metricrows/curves/four5000pairedrefs/mechanisms/efficiency copied3838files.11ownunusedweights5303694Bretired by original queue afterconsumers; bestepoch0 anddependenciesheld. Original3423188 andobserver22853 CLOSED; no replays or activeNN. Negative training and repeateddeveloper/test limitations retained. GoalACTIVE_UNMET, one best sufficient.


## 2026-10-06T11:35:25.855066+08:00 current/future reward alignment actual execution

SOURCE reviewPASS/provisional + actual deployed CPU default-exact witness + four GPU2epoch sanityPASS + four full60/840 fits strictbest/last reloadPASS. Original3801966 nowFULL98_BEST; eightfull98 selection and samehead bothnative/5+31attrs/4refs5000 pending. Same completed1789/event-weighted corpus; A/B/GOLA/C1/old4 frozen, C-only MLP119725 vslinear4588 H3/H32. Allfive+2UNMET; one best sufficient. No NN replay/oldqueue restart/dataset move/power-temperature operation. See current_execution_snapshot in state_commit_current_future and ONEHANDOFF.


## 2026-10-06T11:48:34.655819+08:00 sealed four-fit CPU diagnosis

Actual16cached comparisons/32CPU-head forwards/strictreload PASS; allfour last60 TRAIN true utility positive but reused developer98 negative and predictedpositive. Linear4588 also negative; lossdecrease notdecisiongain. All cachedbest0. Frozenparent futurelabels, unknownmasked; not fullvideo/native efficacy. Collected244historyrows/config/completion/plots, no weight/NPZ copies or liveNN/GPUqueries. Original3801966/sole14340 first12:07 unchanged, all5+2ACTIVE_UNMET.


## 2026-10-06T12:30:49.481953+08:00 actual first4 complete full98 acceptance

All4cachedbest0 complete98/49418 and98TXT exactparent each, Cintervention/geometryhold0: no newtrainedgain. Actual12:11 original3801966 and newNN3867094-97 live FULL98_LAST. Same60 endpoints all98 pending, then oneweight bothnative/all5/31attrs/4refs5000. Sole14340/session91871 next2026-10-06T12:45:56.393785+08:00, then240; no replays/liveNNsource edits/cleanup. smoke_onlyTrue isvalidation-split flag, actualsequence/frame limits0; fullinternal notnativeTEST. All5+2ACTIVE_UNMET; Ccache917 encodedfeatures isnotABCraw-input training.


## 2026-10-06T13:14:01.422464+08:00 actual all8 full98 closed and one model locked

Four60/840 fits and eight98/49418 complete. Last H3MLP .695204/H32MLP .701136/H3linear .712244/H32linear .712990 vsGOLA .743279, allfour5000pairedCI negative. Bestall0 exactgrossparent .746126/noCgain. ONEH3MLPbest0 selectedbeforebothnative. Dated12:49 Las shards launch3933618/19/21/22, original3801966 continues; currentformal5pending. Sole14340/session91871 next2026-10-06T13:31:03.047485+08:00, then240. Originalcompletedreport copied only/noNNorCPUreportreplay. No unchanged C917 fit repeated; newA/B needsraw inputs; currentholdaction notbaseline rejection. Owned11cleanupafterconsumers/protecteddeps. All5+2ACTIVE_UNMET/onebest sufficient.


## 2026-10-06T14:17:32.093926+08:00 Current-future complete actual native closure

Four60/840, eightfull98, one epoch0 both245/220703+234/116649 complete. Five78.016094/74.383784/62.072660 +92.866361/70.022381 exactgrossparent, no trainedCgain, all+2false. All31attrs/81rows, four5000pairedrefs/mechanisms/efficiency. Original3801966 andobserver14340/91871 CLOSED. Elevenownunusedheads2539938B retired by original queue; selected/dependencies retained.1998originalartifacts copied/noNNorreportreplay. Newrelation source/syntheticCPU only, realGPU/controller pending. Rawlabel interpretation corrected: IoU==0nonTarget, partialoverlapunknown; no halfGTsupport among zeroIoU. GoalACTIVE_UNMET; onebest enough.


## 2026-10-06T14:38:11.111040+08:00 Candidate relation real sanity OOM before optimizer

Owner4102464/allfourworkers closed14:32:51; soleobserver22912/12671 exit1 after14:35scheduledread. GPU chunks+cat+dedup OOM requested7.55GiB while17.86allocated. Zeroactualupdates/models/fullfits/parity/native; not algorithmic negative. MinimalCPU concat/dedup thenfinalGPUtransfer, same rows/dtypes/order/batch/targets. ActualMemAvailable245681688kB. Failurelogs preserved in failed_pre_optimizer_oom, no weights produced/deleted. Source re-review and changed-v2 sanity pending, no unchangedretry. Goalactiveunmet and previouscompletefive kept.


## 2026-10-06T14:43:51.162887+08:00 Candidate-relation actual progress

One controller4115079 launched2026-10-06T14:39:21.735409+08:00, actual2026-10-06T14:42:54.141487+08:00 stagePARENT_PARITY. Real completedfit receipts4, no newformalclaim. SOURCE samefamilyPASS/provisional; actual saved pretrainedinitial full98 parent parity pending peractual. Four rawABC configs train A/B/C, not backbone/motion, same2576TRAINstates881seq/196devqueries98seq, oracle/budgeted/action/gross alignedloss. One observer34928, next2026-10-06T14:50:26.444747+08:00,240sec thereafter. Originalclosedfive kept; onebest enough/all+2unmet.


## 2026-10-06T14:58:02.208638+08:00 Candidate-relation actual progress

One controller4115079 launched2026-10-06T14:39:21.735409+08:00, actual2026-10-06T14:54:29.616097+08:00 stageFIT_FULL. Real completedfit receipts4, no newformalclaim. SOURCE samefamilyPASS/provisional; actual saved pretrainedinitial98/49418 parent prediction/9decisions parity PASS; realfour60/480 fullfits active. Four rawABC configs train A/B/C, not backbone/motion, same2576TRAINstates881seq/196devqueries98seq, oracle/budgeted/action/gross alignedloss. One observer34928, next2026-10-06T14:58:28.173877+08:00,240sec thereafter. Originalclosedfive kept; onebest enough/all+2unmet.


## 2026-10-06T15:36:16.920341+08:00 Actual raw ABC supervision coverage

Original controller4115079 FIT_FULL, actual observation2026-10-06T15:34:39.423432+08:00; allfour workersalive. Read-only TRAIN2576/VAL196 states, writepairs4115/455; originalC1keep pausegain>.03 only21/0, candidate-level399/44. ExistingfutureIoUs valid, unknownhistory3378/99 masked. These are states/candidatepairs not independent recoveryevents; no NN/newlabels/TEST or livetraining edits. Original full60/480 plus eightfull98 queue continues, one selectedcheckpoint bothdatasets, all5+2 ACTIVE_UNMET.


## 2026-10-06T15:51:00.369876+08:00 Actual prefix-event census

Existing consistent overlapping prefixes, no NN/TEST/newlabels or livetraining edits. TRAIN normal4385/recovered590/rightcensored126 observed episodes; last-arrived query inputs cover1667/134/126 distinctepisodes. Known incoming writepause gains cover6 recovered+7 rightcensored episodes; VAL no such gain events. Query-1 classification doesnot classify currentquery outcome, firstfailurequery may have normalincomingstate. Unknown/ambiguous states separately retained106TRAIN/9VAL. Original full60/480 andeightfull98 queue continues; all5+2 ACTIVE_UNMET.


## 2026-10-06T16:12:21.071460+08:00 Four raw ABC full training fits closed

Allfour60epochs/480 actualupdates, A/B/C changed/nonzero gradients, frozenC1no grads, best+trueepoch60last strictreload PASS. Bothrelationblocks changed. Actual61epoch records each/244 total. Intake only17 closedtext files, no NN/report replay or weightcleanup. Sameowner4115079 actual2026-10-06T16:11:02.556317+08:00 stageFULL98_BEST; originaleightfull98 then SAMEselected native/reference decision continues, soleobserver34928. Newformalmetrics pending; all5+2 ACTIVE_UNMET.


## 2026-10-06T16:38:51.902506+08:00 Saved best decision decomposition

Four storedbest NPZs vs realparent-equivalent epoch0 savedM0, exactsame196 VAL jobs/C1labels, zero new NN/TEST. Actionschanged2/2/4/4; current improved1/1/2/3 queries, allpause0. HighLRrelations admits3 foundcorrect candidates vsM0zero on samequeries; not sustainedonline proof. LowLR utility includes fewer-search costgain, separately reported. Four configs shareseed42; cached selectiondata isnot independentconfirmation. Originaleightfull98 queue continues; all5+2 ACTIVE_UNMET.


## 2026-10-06T17:15:13.178740+08:00 First four complete98 and cached/online bridge

Allfourbest NN98/49418 complete, sameoriginalowner4115079 nowFULL98_LAST, original126715/16/17/18 alive at2026-10-06T16:46:29.570523+08:00. Every previously savedchangedcacheaction joinedtoonline; cached3correct-extra highLRrel becomes1at corresponding onlineframes; someprefixdrift hurts andboybackpack drift helps. basketball lowLRrelation current.6643 vsparent0, future3.4322 vsfrozenlabel.7495. abmotocometurn41 bothcurrent0/wrongwriteproxy0 butfuture.7626→.5463, actualalso.7627→.5462; no uniquegeocausationclaim. NoNN/reportreplay; oldABCfieldschema mistake corrected direct, stderrpreserved. ActualeightGT/nativepending/all5+2 ACTIVE_UNMET.


## 2026-10-06T18:03:22.243684+08:00 Actual four full60 and eight full98 closed; selected-epoch recovery launched

Original all4x60/480 and all8x98/49418 complete. Best relations_lr5 epoch5 .744415 vsGOLA .743279/grossparent .746126, no +2/effect acceptance. Original parent-gated native skip and20weight40537500B prematurecleanup explicitly corrected; raw receipts retained. New owner253188 restores sameprefix5/40 then archivedarray+full98 prediction/state parity, then actual samehead two native datasets/five/31attrs/four5000CIs. No originalfull60 repeat/newTESTselection/seedstableclaim. Launch only, all newmetrics pending. Protectedold4/C1/GOLA/motion retained, no data move/no hardwarepower-temp. GoalACTIVE_UNMET.


## 2026-10-06T18:22:27.679776+08:00 Actual selected-epoch reconstruction and full98 parity PASS; native starts

At18:17:08 exactall98/49418 TXT+9fields pass; no originalweight-byte identity claim. Actual18:19:59 owner253188 ALIVE/newnativeLASHER_SHARDS start18:17:30 four278791-4; bothdataset4x64 sanity passed. Samepreselectedepoch5 all245/220703+234/116649 forthcoming, not newfiveaccuracycomplete. Sole40544 next2026-10-06T18:49:30.452558+08:00 then240/nearETA, no repeats/earlycleanup. GoalACTIVE_UNMET.


## 2026-10-06T21:08:01.816466+08:00 Selected raw ABC epoch5 complete native evaluation

Original4x60/480 intact; restoredprefix5/40+archivedcached/full98parity PASS; samebest bothfull245/220703 and234/116649 actualGT five/31attrs/curves/four5000pairs COMPLETE. Actualallfive+2=False. No NN/report replay bycollector/publisher; best protected, owninitial/last cleaned onlyafterconsumers. Scope currentrelations/querypause, nottrustedstate/fullhistoricalrepair.
