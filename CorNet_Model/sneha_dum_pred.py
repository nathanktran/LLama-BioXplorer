import sys

def main(abstract_text, title):
    # Simulate processing and return a dummy prediction
    return f"Processed: {abstract_text} / {title}"

if __name__ == "__main__":
    abstract_text = sys.argv[1]
    title = sys.argv[2]
    result = main(abstract_text, title)
    print(result)
