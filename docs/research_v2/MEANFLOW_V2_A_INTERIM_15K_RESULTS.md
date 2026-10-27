# MFV2-A 15K中期评价：seq08/skip10041帧

这是用户指定的最新完整保存EMA快照：7500成功更新、15000成功曝光、2次AMP Skip。不是最终20K结果；对照FM/原MeanFlow均训练20000曝光，因此下表是不同预算的描述性比较，不是同预算提升结论。

相同41帧、相同Source seed、原始GT/ROI、threshold0.5、NFE1/2/4/8。输出为Stage1粗Parent中心，无Stage2、无Jitter、无GT引导推理候选。未选择或调整阈值。中期评价独立脚本保留最终20K严格检查，原训练源码不变。

|模型|曝光|NFE|P %|R %|F1 %|Parent IoU %|Fine-GT Ceiling %|Parents|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|Original FM|20000|1|77.1499|47.8583|58.6172|41.6001|56.5898|110697.6|
|Original FM|20000|2|71.3524|46.5523|55.8945|38.8973|54.9718|116295.8|
|Original FM|20000|4|65.1851|46.0038|53.4790|36.5897|54.0100|125470.8|
|Original FM|20000|8|61.5343|45.1400|51.6079|34.8580|52.8606|130302.0|
|Original MeanFlow|20000|1|78.3682|41.6730|53.9785|37.1116|50.3970|93855.1|
|Original MeanFlow|20000|2|66.3312|40.7321|50.0064|33.4315|48.9066|108132.8|
|Original MeanFlow|20000|4|63.7431|41.3063|49.6607|33.1123|49.1091|114211.3|
|Original MeanFlow|20000|8|60.8942|41.9361|49.1796|32.6802|49.4759|121410.3|
|MFV2-A INTERIM|15000|1|79.8213|41.7214|54.4195|37.5224|50.5337|93280.8|
|MFV2-A INTERIM|15000|2|71.4223|43.4242|53.6028|36.7255|51.7921|108212.5|
|MFV2-A INTERIM|15000|4|64.2606|43.0699|51.1535|34.4524|50.9490|119020.3|
|MFV2-A INTERIM|15000|8|59.2047|43.5537|49.7385|33.1758|51.1024|130475.0|

|模型|曝光|NFE|CD m|IoU0.5 %|IoU0.2 %|IoU0.1 %|JSD-BEV|JSD-3D|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|Original FM|20000|1|0.339895|44.1447|20.1300|3.1281|0.362697|0.470836|
|Original FM|20000|2|0.278264|44.9249|19.1327|3.0229|0.339391|0.476375|
|Original FM|20000|4|0.288182|43.9183|18.2941|2.9387|0.338487|0.491526|
|Original FM|20000|8|0.308620|42.9934|17.6226|2.8609|0.343536|0.503284|
|Original MeanFlow|20000|1|0.333522|40.5748|18.0925|2.8549|0.382696|0.487854|
|Original MeanFlow|20000|2|0.324779|39.5869|16.7901|2.7268|0.370433|0.510815|
|Original MeanFlow|20000|4|0.320598|40.6178|16.6874|2.7109|0.361137|0.512230|
|Original MeanFlow|20000|8|0.331587|40.8147|16.5725|2.7046|0.358446|0.517554|
|MFV2-A INTERIM|15000|1|0.335863|41.3862|18.2342|2.8706|0.376555|0.482405|
|MFV2-A INTERIM|15000|2|0.290885|42.8697|18.1165|2.8847|0.351471|0.487331|
|MFV2-A INTERIM|15000|4|0.310959|41.9045|17.2946|2.7979|0.351489|0.504194|
|MFV2-A INTERIM|15000|8|0.334382|41.2516|16.8910|2.7656|0.354147|0.517090|

完整两档Fixed-K P/R/F1/IoU/Ceiling均在`macro_comparison.csv`。逐帧原始指标保存在压缩JSONL；不以部分帧代替全41帧。A/B正式训练继续，最终20K结果和B结果仍为NOT TESTED；本次不增加训练预算或更改筛选门槛。

GPU2评价显存受6GiB PyTorch allocator上限约束，只与本项目A共享；GPU0/1的其他用户任务不参与。共卡期间不作正式时延/加速宣称。

## Interim interpretation

NFE1 Parent IoU37.5224%, Recall41.7214%, Fine-GT Ceiling50.5337%. Against originalMF20K, IoU+0.4108pp, Recall+0.0484pp, Ceiling+0.1368pp; CD0.335863m versus0.333522m, about0.70% worse. Fixed-K84963 IoU35.3270% versus34.8682%; Fixed-K12000039.0828% versus38.6637%. These are small NFE1 changes, not a demonstrated breakthrough.

NFE2 IoU36.7255% versus oldMF33.4315%, +3.2940pp; CD0.290885m versus0.324779m, about10.44% lower. It still trails originalFM20K NFE2 IoU38.8973% /CD0.278264m. OriginalFM20K NFE1 IoU41.6001% and Ceiling56.5898% remain above this interim candidate.

No same-budget superiority or final GO is claimed. A has15000 exposures, references20000; wait for the registered final20K evaluation. Paired contiguous-frame bootstrap is a descriptive scene-sampling interval only, not a matched-budget or cross-training-seed confidence statement. No extra seed,900-frame evaluation or training extension was started.
