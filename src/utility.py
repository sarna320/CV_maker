import json

def extract_secret(key):
    file_path = './.secret.json'
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            value = data.get(key)
            return value
    except FileNotFoundError:
        return "File not found. Please check the file path."
    except json.JSONDecodeError:
        return "Error decoding JSON. Please check the file content."

if __name__=="__main__":
    print(extract_secret("OPENAI_API_KEY"))