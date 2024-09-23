from cambrian.train.train_fsdp_multi import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
