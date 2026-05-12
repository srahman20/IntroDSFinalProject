# Dataset Instructions

Place the SMS Spam Collection Dataset in this folder.

Supported filenames:
- `SMSSpamCollection`
- `SMSSpamCollection.txt`
- `sms_spam.csv`

The project expects two columns:
- `label`: spam or ham
- `message`: SMS text

If using the original UCI file, it usually has tab-separated values like:

```text
ham    Go until jurong point...
spam   Free entry in 2 a wkly comp...
```

The code automatically handles the common UCI tab-separated format.
