# 2021-11-09 Implementing Python BSC properly
* [DONE] Set up Python Implementation.
* [TODO] from 2021-11-08: Review sampling strategy of second period window and determine if I should do sampling based on activity but following distribution. Remember to set random seed in sampling.
# 2021-11-08 Planning
* [DONE] Plan out file structure for augmented timelines - perhaps writing to extra json files. This would help with processing later, e.g. vectorizing
* [DONE] Clean code and commit
* [DONE] Review follower collection and why it's so slow.
* [TODO] Review sampling strategy of second period window and determine if I should do sampling based on activity but following distribution. Remember to set random seed in sampling.
* [DONE] Initiate Vectoriser for 3-grams and schedule writing csv.
* [DONE] Set up ARC resources.

# 2021-10-21 Meeting with Chico

* Current state of where I am
	* Two stages of data collection, Twitter data limits
	* Plan A: with data from Twitter engineer, can do full analysis.
	* Plan B: without data from Twitter engineer, collect sample in Nov for periods 2 and 3, spend the time until then setting up the feature extraction and statistical analysis.
	* Problems with server space.
	* applying to ARC, waiting on that to come back.
	* politics and compustational social science?
	* match users on followers ('diet')
* Prospective Timeline
	* Meeting with Phil 13 Dec - could you come in person?
	* Idea is then to have a rough draft by that meeting.
	* Then refine over the break until January.
	* Confirmation in Hilary, interview sometime end of HT
* Plan for analysis:
	* Incorporate both retweets and language convergence -> that's what BSC was about
	* However, in my pilots there wasn't much interaction between the users who were assigned the same cluster.
	* 
* Supervision arrangement going forward?
	* would every two weeks suit?
* communication
	* whatsapp? email?


# 2021-10-21
* The following error on self.mapping = np.array in the generate user hashtag matrix script:
File "/home/hubert/DPhil_Studies/2021-04_Study_A_Diffusion/src/d02_intermediate/./generate_user_to_hashtag_matrix.py", line 261, in fit
    self.mapping = np.array(self.mapping)
numpy.core._exceptions.MemoryError: Unable to allocate 265. GiB for an array with shape (133632115,) and data type <U532

* Try to rerun without that?

# 2021-09-20
* Started collection of timeline data of largest block in 2017-2018.
* see 2021-08-03:
	* Things to control for in statistical analysis:
	* activity, average tweeting per day.
	* usage of certain 'neutral' phrases, like 'tout le monde'
	* matched users
	* really useful to have matched users in tracing something up to a point in time.
		* Null hypothesis is not just a flat line. But what kind of a drop off?
		* Generate expectation based on matched users. Tenure, level of activity (on Wiki)
		* just significant activiation in peak in subsequent. compared to matched userset
		* just as I don't have to mass media or prior twitter or social media influences
	* burstiness - Barabasi (read) - Nature paper. Also his book - Bursts. Need to take this into account
		* don't expect sometime to pick up a term and S curve. But it's bursty. Power law. Fat-tailed. Need to expect that. Probably a lot of log transform. Statistically difficult, however.

# 2021-09-03
* Plotting successfully done. More or less the right peaks for each one.
* Now to generate an estimate of the number of users to collect for.

# 2021-09-02
* FInding peaks:
	* arrays of proper length collected
	* trying with different parameters.
	* وأنا كمان has no peaks detected on either base find_peaks or the prominence = 0.6.
	* next: grid search? how to determine which peaks to find?

# 2021-09-01
Next steps:
* find peaks from FAS data
* collect users associated with each peak

# 2021-08-03
Things to control for in statistical analysis:
* activity, average tweeting per day.
* usage of certain 'neutral' phrases, like 'tout le monde'
* matched users
* really useful to have matched users in tracing something up to a point in time.
	* Null hypothesis is not just a flat line. But what kind of a drop off?
	* Generate expectation based on matched users. Tenure, level of activity (on Wiki)
	* just significant activiation in peak in subsequent. compared to matched userset
	* just as I don't have to mass media or prior twitter or social media influences
* burstiness - Barabasi (read) - Nature paper. Also his book - Bursts. Need to take this into account
	* don't expect sometime to pick up a term and S curve. But it's bursty. Power law. Fat-tailed. Need to expect that. Probably a lot of log transform. Statistically difficult, however.


Thoughts
* Since last time
	* created better plots
	* corrected clustering implementation which will take significantly longer, which is fine
	* working on network viz. I think I can get distance measure on the connections between users/phrases across clusters that will exist. The more connections the closer in 2d space.
	* But need to filter out nodes because there are just way too many to plot meaningfully. thinking I will drop the tail end of frequency within each cluster and plot.
	* full archive search collection is still happening. Up to early 2020 now, want to get to Oct 2020 for a neat 3 years.
* Phil mentioned about Twitter person possibly being in contact about data dump, but no response from Phil yet.
* Question: for any qualitative analyses I want to pick out, I'll probably need someone to consult on with the phrasing. What's the best way to go about doing this? I think Siân would be excellent, for example.
* need longer than a week. need longer timespans, maybe fewer users
* need to somehow incorporate interaction anaylsis. Seems clustering on language usage is not often returning internal retweets. Need to check this
* Also interesting that within clusters retweets are not common. Perhaps I need to take this into account directly more.

# 2021-07-21 Logs
* Realisation: even if #metoo itself is only contained within one cluster, a lot of ngrams containing it will appear in other clusters.

# 2021-07-12 Problems with python implementation
* in order to get a decent graphviz we want a graph object. Right? But the R implementation obviously doesn't allow for that.
* But it might - there are matrices involved in there for sure, and how can we visualise these?
* uhtMat is the last matrix object that is then kmeans clustered. I can change what the outputs of these are.
* Idea 2021-07-13: take matrix out of function and then visualise using that matrix.
* Q: how is it that there are only 832 items/phrases?
* THE FUCKING GRAPH ADJACENCY MATRIX THEY HAD DID NOT TRANSFER EDGE WEIGHT
* python: getting a NaN warning from sklearn. Does input need to be dense?
	* nope.
	* perhaps need to drop rows that are 0?
	* adding small diagonal can help better condition the matrix https://stackoverflow.com/questions/18754324/improving-a-badly-conditioned-matrix

* see this for bipartite viz: https://stackoverflow.com/questions/60100006/visualize-bipartite-network-graph-created-using-pandas-dataframe

* also, how to convert between unipartite and bipartite graph type in R:
	* just add a $type attribute

* also, error and warning capture included now for bispec search

# 2021-06-30 Graph Visualisation
* The key problem here is that the graph, when partititioned, does not have multiple assignment. So there will be no overlap in users and phrases assigned across clusters, and a visualisation should have some kind of distance measure. What to do with this?
	* Yesterday I found https://github.com/Nicola17/High-Dimensional-Inspector
	* Perhaps links between users and phrases across clusters, those will still exist
	* but still need to make appropriate adjacency matrix after clustering. How to do?
	* the clustering _is_ a form of rearranging rows of an adjacency matrix. So perhaps don't use the R implementation? Found a python one, but was written in Python 2. need to check if still working.
	* from [@steinbockCasualVisualExploration2018], Common strategies to visualize large graphs – and large data sets in general – are ﬁltering (i.e., removing items) and aggregation (i.e., grouping items)
	* https://scikit-learn.org/stable/modules/biclustering.html biclustering is implemented in sklearn!!!


# 2021-06-21 meeting with Balazs

* FAS collection used this month - 10m for 6 months of tweets.
* BSC results up and running, I can send you the files
* Things suggested to work on:
	* I now have a couple months of FAS on a fuller set of collection data
	* I can start more systematically looking through accounts with clustering information to (a) control for word use to decode who is an activist or not
		* (b) prepare sampling of users for July collection.
---
* use individal plots of course
* strength of dyad
	* follow collaborator
	* normalise histories where a bot was introduced and where a bot was not introduced.
	* counterfactual comparison
	* marginals like account same age, collaboration intensity, normalise out seasonality dynamics, approximate true effect of introduction of bot.
	* seasonality probably not relevant here.

* we do expect an increase across metoo but activist should but compared to what?
* DO VISUALISATION on SPACE for sense of regions
	* correspondence analysis
	* pca, non metric muiltidimensional scaling MDS, correspondnece (bipartite)
	* e.g. measure distance from one cluster to another cluster
	* same measure generating clusters can be used to get distance

* baselines comparisons IMPORTANT!
	* need COUNTERFACTUALS

* distance metric between clusters!
* hashtag pool use
* hashtag cofrequency
* aggregate cofrequency link between clusters

# 2021-06-18
  - [ ] Plot usage of phrases across timeframe
  - [ ] Basic metrics of interaction between users in a cluster in the given timeframe.
  - [ ] How to subsequently model the diffusion
  - [ ] How to scale up to full data collection??
	  * Just do FAS collection first?
	  * How to sample users?
	  * Limiting parameters?
  - [x] enumerate the hashtags to be collected in the FAS for June quota.
  - [x] fix FAS pipeline to use twarc
  - [x] start collection

# 2021-06-17
* Anatomy of a tweet jsonl
	* is a dict
	* **The includes section is all the referenced tweets from the core data bit**
	* structure:
		* data (list)
		* includes (dict)
			* users (lista)
			* tweets
			* polls
			* media
		* meta
			* newest_id
			* oldest_id
			* result_count
			* next_token
		*_twarc
			* url
			* version
			* retrieved_at

# 2021-06-14, 2021-06-15, 2021-06-16
* BSC eval and graphs are working fine. Now working on some of the stuff relating to the 2021-06-08 TODOs
* Realisation after checking the BSC pdf results: hashtags collected and used in vocab also count non-hashtag use of them.
* Also TODO:
  - [x] Fix https removal
	  - [x] attempt made at 15.09 to replace regex with ''\shttps?:\/\/\S*"
  - [x] Fix ellipses removal 
  - [x] Add method to BSC class to generate top words per cluster exmaination
  - [x] merge users and summaries bsc attributes to allow for sorting of phrases by usage.
	  * not sure what I meant by this
  - [x] properly do vocab to not pick up other instances or hashtags
  - [x] Omit clusters with only one user
  - [x] plot histogram of cluster sizes
  - [x] Put data into a networkx graph?
	  - [x] as of today, not worth it. It would be better to figure out how to skip the csv writing stage and just get bsc R script to just handle an adjacency matrix.
  - [x] Add title to ggplots
  - [x] Allow BSC class to store points to collect clusters of interests
  - [x] Allow BSC class to generate list of users and stats based on selected clusters
  - [x] stop hashtag removal in the R script.
  - [x] plot distribution of clusters by users
  - [x] Simply dont count the ngrams crossing tweets by using a custom analyzer
	  * this turned out to be able to massively simplify the CountVectorizer process actually!
	  * No need to do double pass (testing as of 2021-06-16 night) because the custom analyzer can simply drop all the unigrams that aren't hashtags, and then they won't be included in the vocabulary built.
	  - [x] still need to drop the actual eottoken 
  - [x] zoom in on a cluster of users 
  - [x] clear out #................................. stuff fromvocab
		* '#......' is not in the hashtag set of vocabulary, because it's an encoding error with ggplot writing to pdfs (and plots more generally probably)
  - [x] suppress plotnine warnings
  - [x] write comments for 2021-06-16 changes to analyzer
	  - [x] fixed not actually collecting hashtags

# 2021-06-08, 2021-06-10

* Today I am trying to first get rid of the bug where the vocab is not properly rendering
  * CountVectorizer is not properly counting eot_token for example. Confirmed when tokens (generated by the first CountVectorizer instance, remember!) for a column corresponding to an eot_token.
  * Tried lowercasing the vocabulary but that didn't seem to work.
  * Tried setting lowercase to True in second CountVectorier and also removing < and > surrounding eot_token
  * It looks like from https://stackoverflow.com/questions/24007812/can-i-control-the-way-the-countvectorizer-vectorizes-the-corpus-in-scikit-learn?rq=1 that the issue is the second vectorizer needs to have ngram_range too. Going to re-add < > around eot_token and remove lowercase=True from the second CountVectorizer.
  * this should have worked
* Next, work on BSC R script. Desiderata:
  - [x] apply NMI in python search script (normalised mutual information) like the BSC script has
    * see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html
  - [x] How to get the material I want into Python from R
  - [x] How do I want to visualise these?
* Also TODO
  - [x]  Resolve issue about bsc warnings about non-english labelling errors (encoding error)
    * error has been suppressed, in util.R
  - [x] 'null device' printing also suppressed from dev.off()
  - [x] attempted to make parallel the bsc gridsearch but all subprocesses are already parallel, make sense!

# 2021-06-09

* BSC eval is working, but original work had (I think) user NMI score but we can probably also do the same with phrase assignment which will also be useful measure
  * working on this early aft
* now need to add material on getting the max index kind of thing.
- [x] Adding comments to BSC functions

# 2021-06-03

* after the last batch of bsc runs I FINALLY have the eot_tokens and rt tokens removed, but clearly there is still more to clean.
* TODO: comment, clean code, commit changes
  * have cleaned up countvectorizer a bit. Fixed list.extend method because it's an inplace method
* TODO: implement collecting hashtags in vocabulary too.
  * done.
* TODO: further cleaning
  * trying to remove ellipses with regex in the iterator phase so it doesn't need to be processed at all.
  * some error introduced because of the vocab building, inserting and <extend> method into the vocab list FIXED
  * remove URLs too
  * remove user mentions - this can be a separate piece of information to collect
* deleted the archive postprocess script, it's useless now I'm not saving to csv

# 2021-05-24

Since we last spoke, I have

* set up pipeline so far:
	* familiarised myself with the new Twitter v2 API
	* working collection with full archive search.
	* searching the English and the French main hashtags returned roughly 2.5m tweets
	* Have set up pipeline to collect tweets with flexible search query
	* the timelines, however, will be what blows up the rate limit.
	* I've stopped at 8m total collected tweets so I can still use 2m more or so
	* also found out that the user timeline scraping is totally inadequate. Twitter's default endpoint for this gives you the most _recent_ 3200 tweets, but of course I want to be able to specify when exactly that is
	* so I have devised a full archive search that can specify from which users
	* I sorted users who participated most in the English and French campaigns and downloaded the timelines of 2500 users there.
* spent a while debugging and constructing the vocabulary extraction. I thought it would be worse but have figured out a more efficient way to do it without parallelisation or anything like that.
	* cleaned out eot_tokens.
	  * correction: was going to do this but it would simply by much faster to check which mappings have eot_token in them and just skip in writing to csv file for clustering
	  * corrected eot token cleaning
  * TODO: I'm going to have to sit down and think more about data cleaning, removing tokens like 'rt' for retweets and handles. I'll look up existing packages and tools for this. 
* Clustering:
	* adapted some of the original code for this and am working to skip a current major bottleneck of needing everything in a csv file.
	* Writing this csv file takes a lot of time
	* Have set up a search over the parameters relevant for the clustering
	* TODO: next step in the clustering is to exmaine the word clusters, users involved, tweets. Need to set up a handy way to examine these.

* Anna George put me in contact with her previous supervisor.
	* Had a good talk, willing to help with other stuff.

---Meeting---
* random sample?
	* sample the user ids I then collect
* timeline way of deciding if someone is an activist?
* paper idea: more general discussion of how people get into activism?
* nice thing about this is a brand new hashtag.
	* not like environmental activist
* control for prior engagement> did the use the word gender say

NEXT -> see some pilots

# 2021-05-19 Log

* Got bi-spectral clustering working
* Need to be careful in general about the processing of hashtags in the R script provided by the original authors
- [ ] for future, include method to include hashtags as well as phrases.
* MUCH BETTER EFFICIENCY FOUND: concatenate user tweets and use those as 'documents' for the CountVectorizer, rather than doing a whole new pass
* this eliminates much of the inefficiency and kerfuffle with ProcessPoolExecutor.
* And allows much easier expansion to larger user numbers
* Further, I can probably circumvent the original csv requirement for bi-spectral clustering.


* things to do:
  * develop search across number of clusters?

# 2021-05-13 Log

* In the morning about a million tweets had been collected, currently sitting at 7.6m/10m quota. Roughly 2000 users collected, may be enough for a first pass at clustering. We'll see
* 2576 users collected
* only 2490 files downloaded from server though.

# 2021-05-12 Log

- [ ] conversation ids

## Conversation IDs

Conversation IDS: Replies to a given Tweet, as well as replies to those replies, are all included in the conversation stemming from the single original Tweet. Regardless of how many reply threads result, they will all share a common conversation_id to the original Tweet that sparked the conversation.

* Thought: how relevant is this for my work?
  * possibly conversation id for tweets from users to fill out an interaction with another activist that would have been missed otherwise because of the limit imposed on individual user collection.

* in the afternoon: after some fiddling with simply collecting the conversation ids present in user timelines (which are only in the two week span surrounding the oct 17 2017) gives over a million conversation ids, which could easily bust the 10 million monthly cap all on its own.

# 2021-05-11 Logs

* [X] need to add options for datetime
* [ ] idea: implement collecting conversation ids?
* [ ] implement usage monitoring emails


## Advice

N.B. from Ryan Gallagher:

The Twitter API v2 for academics makes it a lot easier to collect complete event data

I currently have a 4 step process that I finally successfully ran end to
end on a real event this week, so I thought it might be useful to share

**1. ** Try to use the filter stream to get as much of the data as possible

I think the 1% threshold still applies here (you can only stream all
filtered tweets if they're <1% of all tweets at any time). If this is
really trouble for you, you can just use the next step

**2. ** Use the full archive endpoint to backfill the early event data that you missed

This is slightly less comprehensive than streaming. Streaming matches
your filters on both a tweet's text AND a quoted tweet's text. The
search only matches to a tweet, not any quoted tweets

Also the completeness of your data will vary based on how quickly you're able to run the full archive search

Note: this is where you need to have the academic track

**3. ** Use all of the conversation IDs of all the tweets from the streaming and the search to get all their reply threads

This is new with v2! There's no excuse anymore not to actually have the conversations that happen around events in your data

**4. ** (optional) Collect the user timelines of those in your dataset

This is so that you have contextual data on what people were saying
prior to a particular event. When we focus just on event data, we lose
the bigger picture

But a warning:

Limit how much of the timelines you collect. If you try to get all 3,200
tweets for every user, you'll sky rocket past the 10,000,000
tweets/month cap

I'm currently trying to go just 2 weeks back for every user. This is the last thing I'm running so we'll see if I hit the cap

So if we use Twitter API v2 in this way, what does this give us

**1) ** complete event data, including early trending that is missed before something goes viral

**2) ** full conversations around the event

**3) ** contextual timeline data so we can situate the event in a broader setting


One potential issue is that there isn't a way around with v2 is the
10,000,000 tweets/month cap. So this pipeline is only going to work on
moderate sized events. Nothing something large or ongoing like the
presidential election or national vaccine rollouts


My code is a bit adapted to our lab machine so it probably isn't
particularly useful, but I hope this thread is helpful for thinking
through how to use the Twitter API v2 for event data

Also, totally forgot the thread numbers. Time for the weekend
