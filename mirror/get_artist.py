import os
import random

artists = [
    "Albert Marquet",
    "Alice Neel",
    "Alphonse Mucha",
    "Amedeo Modigliani",
    "Armand Guillaumin",
    "August Macke",
    "Audrey Kawasaki",
    "Berthe Morisot",
    "Claude Monet",
    "Clive Barker",
    "Diego Rivera",
    "Ed Mell",
    "Edvard Munch",
    "Edmund Dulac",
    "Edward Hopper",
    "Erin Hanson",
    "Fernando Botero",
    "Ferdinand Hodler",
    "Frederick Arthur Verner",
    "Georgia O'Keeffe",
    "Henri Rousseau",
    "Henri de Toulouse-Lautrec",
    "Jacek Yerka",
    "Johannes Vermeer",
    "Louis Comfort Tiffany",
    "Marianne North",
    "Pablo Picasso",
    "Paul Cézanne",
    "Paul Corfield",
    "Paul Gauguin",
    "Paul Signac",
    "Rene Magritte",
    "Rob Gonsalves",
    "Roger Dean",
    "Salvador Dalí",
    "Sandro Botticelli",
    "Scott Naismith",
    "Simon Stalenhag",
    "Syd Mead",
    "Tom Thomson",
    "Umberto Boccioni",
    "Vincent van Gogh",
    "Vladimir Kush",
    "Walter Crane",
]


def sample_artist():
    return random.choice(artists)


def sample_sentence():
    sex = random.choice(["woman", "man"])
    sentences = [
        # "Santa Claus",
        f"a {sex}",
        # f"a {sex} dancer",
        # f"a {sex} hacker",
        # f"a {sex} hiker",
        # f"a {sex} child",
        # f"a {sex} magician",
        # f"a {sex} elf",
        # f"a {sex} student",
    ]
    background = random.choice(
        [
            "flowers",
            "a forest",
            "a beach",
            "a cloudy sky",
        ]
    )
    return f"Mesmerizing oil painting of {random.choice(sentences)} in front of {background}, "

# def sample_sentence():
#     sex = random.choice(["woman", "man"])
#     sentences = [
#         f"a {sex} with a tropical outfit",
#     ]
#     background = random.choice(
#         [
#             "a tropical forest",
#             "a tropical beach",
#             "a tropical environment",
#         ]
#     )
#     return f"Mesmerizing oil painting of {random.choice(sentences)} in front of {background}, "
