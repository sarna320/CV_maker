1. Clone repository
```sh
git clone https://github.com/sarna320/CV_maker
```

2. Make venv
```sh
python3 -m venv .env
```

```sh
source .env/bin/activate
```

Install requirments.txt.
```sh
pip install -r requirements.txt
```

Additional commands:
```sh
pip freeze > requirements.txt
```

```sh
deactivate
```

3. Make .secret.json
```sh
touch ./.secret.json
```
```sh
echo '{"OPENAI_API_KEY": "your_openai_api_key"}' > ./.secret.json
```

4. Test extraction agent
```sh
python3 src/extract.py 
```