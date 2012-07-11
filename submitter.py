from Corp import stop_words, Files, Corp
from gensim import corpora, models, similarities
import logging
import json
import cPickle
import random
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# First, we make a dictionary of words used in the posts
"""
myFiles = Files([open("../trainPosts.json"), open("../testPosts.json")])

dictionary = corpora.Dictionary(doc for doc in myFiles)
stop_ids = [dictionary.token2id[stopword] for stopword in stop_words if stopword in dictionary.token2id]
infreq_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq < 50]
dictionary.filter_tokens(stop_ids + infreq_ids) # remove stop words and words that appear infrequently
dictionary.compactify() # remove gaps in id sequence after words that were removed

dictionary.save("dictionary.saved")
"""
dictionary = corpora.dictionary.Dictionary.load("dictionary.saved") 

# Next, we train the LDA model with the blog posts, estimating the topics
"""
myCorp = Corp(myFiles, dictionary)
lda = models.ldamodel.LdaModel(corpus=myCorp, id2word=dictionary, num_topics=100, update_every=1, chunksize=10000, passes=1)

lda.save("lda.saved")
"""
lda = models.ldamodel.LdaModel.load("lda.saved") 

#myFiles.close_files()

# Now, we do some quick preliminary work to determine which blogs have which posts, and to map post_id's to a zero-based index, or vice versa

TrainPostIndeces = {}
blogTrainPosts = {}
f = open("../trainPostsThin.json")
i = 0
for line in f:
    post = json.loads(line)
    blog_id = post["blog"]
    post_id = post["post_id"]
    TrainPostIndeces[post_id] = i
    if not blogTrainPosts.has_key(blog_id):
        blogTrainPosts[blog_id] = []
    blogTrainPosts[blog_id].append(post_id)
    i += 1
f.close()
print "Done doing preliminary training data processing"

TestPostIds = []
TestPostIndeces = {}
blogTestPosts = {}
i = 0
f = open("../testPostsThin.json")
for line in f:
    post = json.loads(line)
    blog_id = post["blog"]
    post_id = post["post_id"]
    TestPostIds.append(post_id)
    TestPostIndeces[post_id] = i
    if not blogTestPosts.has_key(blog_id):
        blogTestPosts[blog_id] = []
    blogTestPosts[blog_id].append(post_id)
    i += 1
f.close()
print "Done doing preliminary test data processing"

# We build a lookup-index of test posts, for quick answers to questions about what test posts are similar to a given training post
"""
myFilesTest = Files([open("../testPosts.json")])
myCorpTest = Corp(myFilesTest, dictionary)
TestVecs = [vec for vec in lda[myCorpTest]]
TestIndex = similarities.Similarity("./simDump/", TestVecs, num_features=100)
TestIndex.num_best = 100
myFilesTest.close_files()
"""

#cPickle.dump(TestVecs, open("TestVecs.saved", "w"))
TestVecs = cPickle.load(open("TestVecs.saved", "r"))

#TestIndex.save("TestIndex.saved")
TestIndex = similarities.Similarity.load("TestIndex.saved")
print "Done making the test lookup index"

# We estimate the training topics, which we can hold in memory since they are sparsely coded in gensim
"""
myFilesTrain = Files([open("../trainPosts.json")])
myCorpTrain = Corp(myFilesTrain, dictionary)
TrainVecs = [vec for vec in lda[myCorpTrain]]
myFilesTrain.close_files()

cPickle.dump(TrainVecs, open("TrainVecs.saved", "w"))
"""
TrainVecs = cPickle.load(open("TrainVecs.saved", "r"))
print "Done estimating the training topics"

# Now we begin making submissions
print "Beginning to make submissions"
users = open("../trainUsers.json", "r")
submissions = open("submissions.csv", "w")
submissions.write("\"posts\"\n")
user_total = 0
for line in users:
    user = json.loads(line)
    if user["inTestSet"] == False:
        continue

    blog_weight = 2.0
    posts = {} # The potential posts to recommend and their scores

    liked_blogs = [like["blog"] for like in user["likes"]]
    for blog_id in liked_blogs:
        if blogTestPosts.has_key(blog_id): # Some blogs with posts in the train period might have no posts in the test period
            for post_id in blogTestPosts[blog_id]:
                if not posts.has_key(post_id):
                    posts[post_id] = 0
                posts[post_id] += blog_weight / float(len(blogTestPosts[blog_id]))
    # After this, posts[post_id] = (# times blog of post_id was liked by user in training) / (# posts from blog of post_id in training)
    posts_indeces = [TestPostIndeces[post_id] for post_id in posts.keys()]
    posts_vecs = [TestVecs[i] for i in posts_indeces]

    liked_post_indeces = []
    for like in user["likes"]:
        try: # For whatever reason, there is a slight mismatch between posts liked by users in trainUsers.json, and posts appearing in trainPosts.json
            liked_post_indeces.append(TrainPostIndeces[like["post_id"]])
        except:
            print "Error: Bad index!"

    total_likes = len(liked_post_indeces)
    sample_size = min(10, total_likes)
    liked_post_indeces = random.sample(liked_post_indeces, sample_size) # to cut down computation time
    liked_post_vecs = [TrainVecs[i] for i in liked_post_indeces]
    LikedPostIndex = similarities.SparseMatrixSimilarity(liked_post_vecs, num_terms=100)

    for posts_index, similar in zip(posts_indeces, LikedPostIndex[posts_vecs]):
        posts[TestPostIds[posts_index]] += max([rho for rho in similar])
    # ie, posts[post_id] += max(semantic similarities to sample of previously liked posts) 

    if len(posts) < 100: # Fill up remaining spaces with posts semantically similar to previously liked posts, (almost always from different blogs)
        similar_posts_ids  = [(TestPostIds[i], rho) for similar100 in TestIndex[liked_post_vecs] for (i, rho) in similar100]
        for (post_id, rho) in similar_posts_ids:
            if not posts.has_key(post_id):
                posts[post_id] = 0
            posts[post_id] += rho / float(sample_size) / float(sample_size)

    # Now pick the top 100 blogs, (or less if that's the case)
    recommendedPosts = list(sorted(posts, key=posts.__getitem__, reverse=True))
    output = " ".join(recommendedPosts[0:100]) + "\n"
    submissions.write(output)
    
    if user_total % 100 == 0:
        print "User " + str(user_total) + " out of 16262"
    user_total = user_total + 1
