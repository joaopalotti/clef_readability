import praw
import pandas as pd

reddit = praw.Reddit(client_id='Wbeaou8x1p_JBA',
                     client_secret='xCbEGxCppr_NyREXM9La1QULeGU',
                     user_agent='myuseragent')

print(reddit.read_only)  # Output: True

def get_comments(comment):

    # we need to get more comments...
    if type(comment) == praw.models.reddit.more.MoreComments:
        text = ""
        for other_comment in comment.comments():
            text = text + " " + get_comments(other_comment)

        return text

    text = comment.body
    for child in comment.replies:
        text = text + " " + get_comments(child)
    return text

def get_text(sub):

    text = sub.title
    text = text + " " + sub.selftext

    for c in sub.comments:
        text = text + " " + get_comments(c)

    return text

# https://www.reddit.com/r/medical/
subreddits = ["medical", "AskDocs", "AskDoctorSmeeee", "Health", "WomensHealth", "Mens_Health"]

texts = {}
for subr in subreddits:
    texts[subr] = {}

# Get HOT submissions for period
for subr in subreddits:
    print("Processing %s" % (subr))
    for submission in reddit.subreddit(subr).hot(limit=None):
        texts[subr][submission.id] = get_text(submission)
        print(" -- Submission %s" % (submission.id))

# Get TOP submissions for period
for subr in subreddits:
    print("Processing %s" % (subr))
    for time in ["day", "hour", "month", "week", "year", "all"]:
        for submission in reddit.subreddit(subr).top(time):
            texts[subr][submission.id] = get_text(submission)
            print(" -- Submission %s" % (submission.id))


# Get "all" submissions for period
# Timestamp between 1st Jan 2016 and 31st Dec 2016
# 1451606401 to 1483228799

starttime = 1451606401
endtime = 1483228799
for subr in subreddits:
    print("Processing %s" % (subr))
    counter = 0
    for submission in reddit.subreddit(subr).submissions(start=starttime, end=endtime):
        try:
            texts[subr][submission.id] = get_text(submission)
            print("%d -- Submission %s" % (counter, submission.id))
            counter += 1

        except:
            print("%d -- Exception caught for submission %s" % (counter, submission.id))
            continue


for subr in subreddits:
    print("Len %s: %d" % (subr, len(texts[subr])))


dfs = []
for subr in subreddits:
    df = pd.DataFrame.from_dict(texts[subr], orient="index")
    df["subreddit"] = subr
    dfs.append(df)

merged = pd.concat(dfs)
merged = merged.reset_index()
merged.columns = ["submission","text","subreddit"]

merged.to_csv("reddit.txt", index=False)

