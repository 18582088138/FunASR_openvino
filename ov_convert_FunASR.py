from funasr import AutoModel
# paraformer-zh is a multi-functional asr model
# use vad, punc, spk or not as you need
hotwords = ''
# hotwords = '你的热词 魔搭 脊背发凉 chatGPT 天马流星拳 劈里啪啦 PaLM GPT4 文心一言 一枪 Midjourney'
model = AutoModel(model="paraformer-zh", model_revision="v2.0.4",
                  vad_model="fsmn-vad", vad_model_revision="v2.0.4",
                  punc_model="ct-punc-c", punc_model_revision="v2.0.4",
                  spk_model="cam++", spk_model_revision="v2.0.2",
                  )

# res = model.generate(input=f"{model.model_path}/example/asr_example.wav", 
# res = model.generate(input="BAC009S0764W0133.wav", 
res = model.generate(input="BAC009S0764W0262.wav", 
            batch_size_s=300, 
            hotword=hotwords)
print(res)

model.export()
print("==== export onnx model success =====")