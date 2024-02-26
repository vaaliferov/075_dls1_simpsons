### DLS1_SIMPSONS [:link:](https://stepik.org/course/101721/syllabus)

* model: [EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch) trained on [journey-springfield](https://www.kaggle.com/c/journey-springfield/data)
* deployment: vm instance on [oracle cloud](https://www.oracle.com/cloud)
* demo: https://t.me/vaaliferov_simpsons_bot

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
gdown 1LvWdVCH4qeQIB-K-jKLQDPPgFr5ZOPQd
python3 bot.py <bot_owner_id> <bot_token>
```

![Alt Text](pics/tg.png)