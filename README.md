# TGI Op Optimization Based On FasterTransformer

transit the FasterTransformer op into TGI, learn from the code

ref:https://zhuanlan.zhihu.com/p/680341550

TGI 入口：

logits_process/ TopKlogitsWrapper

FT 入口：

dynamic_decoder_layer 中 forward（logits, output_ids) -> 由beamwidth取值进入不同函数：

sample层位置：
LMHead(linear) -> topk -> FusedSoftmaxAndSampling -> Output_token_ids

topK 核心优化：
第一轮分成 block per beam, 对每个 block per beam 做 topk
第二轮再做一次topk
利用大顶堆维护topk

sampling优化：
Softmax移到后面，这样算的数变少了， top k 放中间有收益
Softmax对数值顺序无影响，不影响topk结果
