###war
# This is based on the original bi-spectral clustering script, modified to take cammand line arguments
###
library(argparser, quietly=TRUE)

# Create a parser
parser <- arg_parser("Round a floating point number")

parser <- add_argument(parser,
  "filename",
  help="The user to hashtag edgelist filename that is required for the clustering"
)

parser <- add_argument(parser,
  "outdir",
  help="The output directory"
)

parser <- add_argument(parser,
  "--min_user",
  help="The minimum number of users a phrase/hashtag must have been used to be included",
  default=10
)

parser <- add_argument(parser,
  "--ncluster",
  help="how many clusters to make?",
  default=100
)

parser <- add_argument(parser,
  "--verbose",
  help = "whether to pring how long clustering took",
  default = FALSE
)

# parse argument
# args <- parse_args(parser,
#   c(
#     "/Users/hubert/Nextcloud/DPhil/DPhil_Studies/2021-04_Study_A_Diffusion/collection_results_2021_05_04_16_22/bispec_ready_counts.csv",
#     "/Users/hubert/Nextcloud/DPhil/DPhil_Studies/2021-04_Study_A_Diffusion/collection_results_2021_05_04_16_22/bsc/",
#     "--min_user", 10,
#     "--ncluster", 101
#   )
# )

args <- parse_args(parser)

# date
report_date <- Sys.Date()

# Where is your user to hashtag matrix? Make sure it is as defined in the Readme!
USER_TO_HASHTAG_EDGELIST_FILENAME <- args$filename
OUTPUT_DIRECTORY <- args$outdir
#Paramters for bispectral coclustering
## What is the minimum number of users a hashtag must have been used by to be included?
MIN_USER <- args$min_user
## How many clusters do we want?
N_CLUSTERS <- args$ncluster

# Where to place the output 
# dir.create(OUTPUT_DIRECTORY)

# load r utils
# print(dirname(sys.frame(1)$ofile))
# source(paste0(dirname(sys.frame(1)$ofile),"/util.R"))
source(paste0(getwd(),"/util.R"))

# You can optionally identify some users (we do this via ID) and hashtags that you are particularly interested in here
known_important_users <- c("29417304","755113","16948493")
known_important_hashtags <- c("charlotteprotests", "keithlamontscott",
                              "charlotte","charlotteriot","charlottepd",
                              "charlotteuprising","blacklivesmatter")

# load the data in
if(grepl("[.]gz$", USER_TO_HASHTAG_EDGELIST_FILENAME)){
  # Note, you might not be able to use zcat here, esp. if on Windows. There are other ways to read in gzipped files though!
  user_ht <- load_user_ht_matrix(paste0("zcat < ", USER_TO_HASHTAG_EDGELIST_FILENAME))
} else {
  user_ht <- fread(USER_TO_HASHTAG_EDGELIST_FILENAME)
}

# run bispectral clustering
## The function takes as input the 
listObj=biSpectralCoCluster(user_ht,min_user=MIN_USER,k=N_CLUSTERS, verbose = FALSE)

# 2021-06-08: save dataframes in bispectral clustering
csv_output_name <- paste0("bsc_", MIN_USER, "_", N_CLUSTERS)
write.csv(listObj$summary, file.path(OUTPUT_DIRECTORY, paste0(csv_output_name,'_summary.csv')))
write.csv(listObj$users, file.path(OUTPUT_DIRECTORY, paste0(csv_output_name,'_users.csv')))
write.csv(listObj$hashtags, file.path(OUTPUT_DIRECTORY, paste0(csv_output_name,'_hashtags.csv')))

# output_filename
output_filename <- paste0("hashtags_per_cluster_",MIN_USER,"_",N_CLUSTERS,"_.pdf")

# look at results
gen_plots(listObj,min_user_count = 1,filename=file.path(OUTPUT_DIRECTORY, output_filename))

# # You can identify a subset of clusters that look interesting after reviewing the output of the above function!
# clusters_of_interest <- c(70,64,57, 34, 9,24, 13,79, 50, 8,47)
# 
# # Pull the users and hashtags from the clustering
# userData = listObj[["users"]]
# userData <- data.table(userData)
# 
# htData = listObj[["hashtags"]]
# htData$hashtag <- str_replace_all(htData$hashtag,"#","")
# htData$hashtag <- tolower(htData$hashtag)
# htData <- data.table(htData)
# 
# # What clusters are our known hashtags of interest in?
# table(htData[htData$hashtag %in% known_important_hashtags,]$topic_cluster)
# 
# # Generate counts of hashtags/users per cluster
# counts <- merge(userData[,.N,by=topic_cluster],data.table(htData)[,.N,by=topic_cluster],by="topic_cluster")
# p1 <- ggplot(counts, aes(N.x,N.y)) + geom_point() + scale_x_log10() + scale_y_log10() + ylab("Number of Hashtags in Cluster") + xlab("Number of Users In Cluster")
# p1
# 
# # Rerun bispectral co-clustering on a subset of our data (as is done in the paper)
# he_13 <- user_ht[Source %in% c(userData[topic_cluster==13]$ID)]
# he_13 <- he_13[Target %in% paste0("#",htData[topic_cluster==13]$hashtag)]
# clustering_13 <- biSpectralCoCluster(he_13,min_user=10,k=25)
# gen_plots(clustering_13,min_user_count = 1,filename=file.path(OUTPUT_DIRECTORY, "cluster_13_sub.pdf"))
# 
# # subset our data to users and hashtags of interest
# important_users <- userData[userData$topic_cluster %in% clusters_of_interest,]
# important_hashtags <- htData[htData$topic_cluster %in% clusters_of_interest,]
# ## Do these clusters contain all users and hashtags we were originally interested in?
# setdiff(known_important_users, important_users$ID)
# setdiff(known_important_hashtags, important_hashtags$hashtag)
# 
# # write out the final list of users and hashtags that might be of interest for further study
# write.csv(important_users, file.path(OUTPUT_DIRECTORY,"important_users.csv"))
# write.csv(important_hashtags, file.path(OUTPUT_DIRECTORY,"important_hashtags.csv"))
# 
# # write out the R data file representing the final clustering
# rm(user_ht)
# save.image( file.path(OUTPUT_DIRECTORY,"final_clustering.rdata"))
# 
# 
# # Code for assessing appropriate cluster size
# # Sensitivity
# library(aricode)
# 
# res <- list()
# for(k in 50:300){
#   r <- biSpectralCoCluster(he_13,min_user = 10, k = k)
#   res[[k]] <- c(r[['users']]$topic_cluster, r[['hashtags']]$topic_cluster)
# }
# 
# rd <- data.frame()
# for(k in 60:300){
#   for(i in 1:10){
#     rd <- rbind(rd, data.frame(k=k,i=i, nmi=NMI(res[[k]],res[[k-i]])))
#   }
# }
# rd <- data.table(rd)
# theme_set(theme_bw(20))
# p <- ggplot(rd[, mean(nmi), by=k], aes(k, V1)) + geom_point() + geom_line() + stat_smooth()
# p <- p + xlab("Number of Clusters") + ylab("Mean\nNormalized Mutual\nInformation (NMI)")
# p




