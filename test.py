from mirror.get_artist import sample_artist, sample_sentence

prompt = (
    # f"Mesmerizing oil painting of a person, "
    f"{sample_sentence()}"
    f"by (({sample_artist()})), "
    f"by ({sample_artist()}), "
    f"by ((({sample_artist()})))"
)
print(f"{prompt = }")
