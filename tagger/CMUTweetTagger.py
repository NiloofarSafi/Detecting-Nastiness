#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple Python wrapper for runTagger.sh script for CMU's Tweet Tokeniser and Part of Speech tagger: http://www.ark.cs.cmu.edu/TweetNLP/
Usage:
results=runtagger_parse(['example tweet 1', 'example tweet 2'])
results will contain a list of lists (one per tweet) of triples, each triple represents (term, type, confidence)
"""
import subprocess
import shlex
# The only relavent source I've found is here:
# http://m1ked.com/post/12304626776/pos-tagger-for-twitter-successfully-implemented-in
# which is a very simple implementation, my implementation is a bit more
# useful (but not much).

# NOTE this command is directly lifted from runTagger.sh

import os

RUN_TAGGER_CMD = "java -XX:ParallelGCThreads=2 -Xmx1g -jar %s" % os.path.join('resources/ark-tweet-nlp-0.3.2/ark-tweet-nlp-0.3.2.jar')


def _split_results(rows):
    """Parse the tab-delimited returned lines, modified from: https://github.com/brendano/ark-tweet-nlp/blob/master/scripts/show.py"""
    for line in rows:
        line = line.strip()  # remove '\n'
        if len(line) > 0:
            if line.count('\t') == 2:
                parts = line.split('\t')
                tokens = parts[0]
                tags = parts[1]
                confidence = float(parts[2])
                yield tokens, tags, confidence


def _call_runtagger(tweets, run_tagger_cmd=RUN_TAGGER_CMD):
    """Call runTagger.sh using a named input file"""

    # remove carriage returns as they are tweet separators for the stdin
    # interface
    tweets_cleaned = [tw.replace('\n', ' ') for tw in tweets]
    message = "\n".join(tweets_cleaned)

    # force UTF-8 encoding (from internal unicode type) to avoid .communicate encoding error as per:
    # http://stackoverflow.com/questions/3040101/python-encoding-for-pipe-communicate
    message = message.encode('utf-8')

    # build a list of args
    args = shlex.split(run_tagger_cmd)
    args.append('--output-format')
    args.append('conll')
    po = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # old call - made a direct call to runTagger.sh (not Windows friendly)
    # po = subprocess.Popen([run_tagger_cmd, '--output-format', 'conll'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result = po.communicate(message)
    # expect a tuple of 2 items like:
    # ('hello\t!\t0.9858\nthere\tR\t0.4168\n\n',
    # 'Listening on stdin for input.  (-h for help)\nDetected text input format\nTokenized and tagged 1 tweets (2 tokens) in 7.5 seconds: 0.1 tweets/sec, 0.3 tokens/sec\n')

    pos_result = result[0].strip('\n\n')  # get first line, remove final double carriage return
    pos_result = pos_result.split('\n\n')  # split messages by double carriage returns
    pos_results = [pr.split('\n') for pr in pos_result]  # split parts of message by each carriage return
    return pos_results


def runtagger_parse(tweets, run_tagger_cmd=RUN_TAGGER_CMD):
    """Call runTagger.sh on a list of tweets, parse the result, return lists of tuples of (term, type, confidence)"""
    pos_raw_results = _call_runtagger(tweets, run_tagger_cmd)
    pos_result = []
    for pos_raw_result in pos_raw_results:
        pos_result.append([x for x in _split_results(pos_raw_result)])
    return pos_result


def check_script_is_present(run_tagger_cmd=RUN_TAGGER_CMD):
    """Simple test to make sure we can see the script"""
    success = False
    # run_tagger_cmd='''java -XX:ParallelGCThreads=2 -Xmx1g -jar E:\\suraj\\cyberbullying\\tagger\\resources\\ark-tweet-nlp-0.3.2\\ark-tweet-nlp-0.3.2.jar'''
    try:
        args = shlex.split(run_tagger_cmd)
        print args
        args.append("--help")
        po = subprocess.Popen(args, stdout=subprocess.PIPE)
        # old call - made a direct call to runTagger.sh (not Windows friendly)
        # po = subprocess.Popen([run_tagger_cmd, '--help'], stdout=subprocess.PIPE)
        while not po.poll():
            lines = [l for l in po.stdout]
            print lines
        # we expected the first line of --help to look like the following:
        assert "RunTagger [options]" in lines[0]
        success = True
    except OSError as err:
        print "Caught an OSError, have you specified the correct path to runTagger.sh? We are using \"%s\". Exception: %r" % (
        run_tagger_cmd, repr(err))
    return success


if __name__ == "__main__":
    import csv
    from preprocess.normalize import apply_rules
    import pandas as pd


    def replace_all(text, dic):
        for i, j in dic.iteritems():
            text = text.replace(i, j)
        return text

    ASKFM = {
        r'[@＠][a-zA-Z0-9_]+': "@username",  # @amb1213 -> @username
        r"((www\.[^\s]+)|(https?:\/\/[^\s]+))": "URL",  # url
        r"#([a-zA-Z0-9_]+)": r"\1",  # remove hashtag

    }

    normalize_askfm = lambda text: apply_rules(ASKFM)(text)
    print RUN_TAGGER_CMD
    print "Checking that we can see \"%s\", this will crash if we can't" % (RUN_TAGGER_CMD)
    success = check_script_is_present()
    new_row = ['pos_tag']
    # new_row1 = ['pos_tag_question']
    # new_row2 = ['pos_tag_answer']
    if success:
        with open('/home/niloofar/Niloofar/Data_Collection/wiki.csv',
                  mode='r') as csvfile:
            reader = csv.DictReader(csvfile)
            headers = reader.fieldnames
            j = 0
            for row in reader:
                s = ''
                # s1 = ''
                # s2 = ''
                i = 0
                for tagged_post in runtagger_parse([unicode(row['comment'], 'utf-8')]):
                    for t,p,c in tagged_post:
                        if i == 0:
                            s = s + "/".join([t, p, str(c)])
                            i = i + 1
                        else:
                            s = s + " " + "/".join([t, p, str(c)])
                new_row.append(s)
                print(j)
                j+=1
                # for tagged_post in runtagger_parse([unicode(row['question'], 'utf-8')]):
                #     for t,p,c in tagged_post:
                #         if i == 0:
                #             s1 = s1 + "/".join([t, p, str(c)])
                #             i = i + 1
                #         else:
                #             s1 = s1 + " " + "/".join([t, p, str(c)])
                #
                # new_row1.append(s1)
                # i = 0
                # for tagged_post in runtagger_parse([unicode(row['answer'], 'utf-8')]):
                #     for t, p, c in tagged_post:
                #         if i == 0:
                #             s2 = s2 + "/".join([t, p, str(c)])
                #             i = i + 1
                #         else:
                #             s2 = s2 + " " + "/".join([t, p, str(c)])
                # new_row2.append(s2)
                # new_row1.append(" ".join(["/".join([t, p, str(c)]) for t, p, c in tagged_post]) for tagged_post in runtagger_parse([unicode(row['question'], 'utf-8')]))
                # new_row2.append(" ".join(["/".join([t2, p2, str(c2)]) for t2, p2, c2 in tagged_post2]) for tagged_post2 in runtagger_parse([unicode(row['answer'], 'utf-8')]))

        my_df = pd.DataFrame(new_row)

        my_df.to_csv(
            '/home/niloofar/Niloofar/all/niloofar/Niloofar/suraj/cyberbullying/resources/wiki_pos_tags.csv',
            index=False, header=False)

        # my_df = pd.DataFrame(new_row1)
        #
        # my_df.to_csv('/home/niloofar/Niloofar/all/niloofar/Niloofar/suraj/cyberbullying/resources/Data_01_26_2016_pos_tags1.csv', index=False, header=False)
        #
        # my_df2 = pd.DataFrame(new_row2)
        #
        # my_df2.to_csv('/home/niloofar/Niloofar/all/niloofar/Niloofar/suraj/cyberbullying/resources/Data_01_26_2016_pos_tags2.csv', index=False, header=False)

        # print "Success."
        # print "Now pass in two messages, get a list of tuples back:"
        # tweets = ['this is a message']

        #print runtagger_parse(tweets)

    # subprocess.Popen(['java', '-XX:ParallelGCThreads=2', '-Xmx1g', '-jar', 'resources/ark-tweet-nlp-0.3.2/ark-tweet-nlp-0.3.2.jar'],stdout=subprocess.PIPE)
