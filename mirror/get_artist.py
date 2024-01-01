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
    sentences = [
        "Mesmerizing oil painting of a person, ",
        "Mesmerizing oil painting of Santa Claus, ",
        "Mesmerizing oil painting of woman and flowers, ",
    ]
    return random.choice(sentences)
