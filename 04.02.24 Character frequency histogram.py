import matplotlib.pyplot as plt

def count_latin_letters(filename):
    letter_counts = {}
    try:
        with open(filename, 'r') as file:
            for line in file:
                for char in line:
                    if char.isalpha() and char.isascii():  # Check if character is a Latin letter
                        char = char.lower()  # Convert to lowercase to treat upper- and lower-case as equal
                        letter_counts[char] = letter_counts.get(char, 0) + 1
    except FileNotFoundError:
        print("File not found.")
    return letter_counts

def create_pie_chart(letter_counts):
    labels = sorted(letter_counts.keys())
    sizes = [letter_counts[letter] for letter in labels]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Letter Distribution')
    plt.show()

filename = input("Enter the name of the input file: ")
letter_counts = count_latin_letters(filename)
if letter_counts:
    create_pie_chart(letter_counts)
