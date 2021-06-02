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