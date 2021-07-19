###war
# This is based on the original bi-spectral clustering script, modified to take cammand line arguments
###
library(argparser, quietly=TRUE)
suppressPackageStartupMessages(library(bit64,quietly=TRUE))
suppressPackageStartupMessages(library(data.table,quietly=TRUE))
suppressPackageStartupMessages(library(igraph,quietly=TRUE))
suppressPackageStartupMessages(library(Matrix,quietly=TRUE))
suppressPackageStartupMessages(library(rARPACK,quietly=TRUE))
suppressPackageStartupMessages(library(wordcloud,quietly=TRUE))
suppressPackageStartupMessages(library(ggplot2,quietly=TRUE))
suppressPackageStartupMessages(library(dplyr,quietly=TRUE))
suppressPackageStartupMessages(library(irlba,quietly=TRUE))
suppressPackageStartupMessages(library(stringr,quietly=TRUE))

# Create a parser
# parser <- arg_parser("Round a floating point number")
# 
# parser <- add_argument(parser,
#                        "filename",
#                        help="The user to hashtag edgelist filename that is required for the clustering"
# )
# 
# parser <- add_argument(parser,
#                        "outdir",
#                        help="The output directory"
# )
# 
# parser <- add_argument(parser,
#                        "--min_user",
#                        help="The minimum number of users a phrase/hashtag must have been used to be included",
#                        default=10
# )
# 
# parser <- add_argument(parser,
#                        "--ncluster",
#                        help="how many clusters to make?",
#                        default=100
# )
# 
# parser <- add_argument(parser,
#                        "--verbose",
#                        help = "whether to pring how long clustering took",
#                        default = FALSE
# )

# parse argument
args <- parse_args(parser,
  c(
    "/Users/hubert/Nextcloud/DPhil/DPhil_Studies/2021-04_Study_A_Diffusion/collection_results_2021_05_04_16_22/bispec_ready_counts.csv",
    "/Users/hubert/Nextcloud/DPhil/DPhil_Studies/2021-04_Study_A_Diffusion/collection_results_2021_05_04_16_22/bsc/",
    "--min_user", 10,
    "--ncluster", 500
  )
)


# args <- parse_args(parser)

# date
report_date <- Sys.Date()

set.seed(0)
biSpectralCoCluster=function(h_edges,min_user=1,k=100,all_hashtags=FALSE,verbose = FALSE){
  
  H=graph.data.frame(h_edges)
  S=simplify(H,remove.loops=FALSE,remove.multiple=TRUE)
  # rm(h_edges)
  A=get.adjacency(H,names=TRUE, attr='weight')
  S=get.adjacency(S, attr = 'weight')
  
  # HUBERT NOTE: THIS NEXT LINE TREATS ENTRIES IN TARGET COLUMN AS HASHTAGS
  if (all_hashtags){
    mapping=grepl('#',V(H)$name)  
  }else{
    mapping=!grepl("^[0-9]+$", V(H)$name)
  }
  
  A=A[,mapping]
  S=S[,mapping]
  # rm(H)
  A=A[!mapping,]
  S=S[!mapping,]
  
  ht_mapping= colSums(S)>=min_user
  A=A[,ht_mapping]
  rm(S)
  
  if (verbose){
    cat('graph constructed')
  }
  
  start = Sys.time()
  Ucount=rowSums(A)
  HTcount=colSums(A)
  d1=1/sqrt(rowSums(A))
  d1[is.infinite(d1)]=0
  D1=Diagonal(n=dim(A)[1],x=d1)
  d2=1/sqrt(colSums(A))
  d2[is.infinite(d2)]=0  
  D2=Diagonal(n=dim(A)[2],x=d2)
  An=D1%*%A%*%D2
  # for(k in seq(from=10,to=500,by=20)){
  #   for (l in seq(from=0, to=1, by=0.05)){
  #     zzz <- matrix(0, ncol=3383, nrow=2363)
  #     diag(zzz) <- l
  #     
  #     print(k)
  #     print(l)
  #     
  #     obj=irlba(An+zzz,k,nu=k,nv=k,maxit = 2000, tol=1e-4, verbose=TRUE, work=2000)
  #   }
  # }
  obj=irlba(An,k,nu=k,nv=k,maxit = 2000, verbose=verbose, work=2000)
  if (verbose){
    print(paste("Bispectral took:", Sys.time()-start,"seconds"))
  }
  
  uMat=data.frame(ID=rownames(A),degree=Ucount,as.matrix(D1%*%obj$u),stringsAsFactors=FALSE)
  htMat=data.frame(ID=colnames(A),degree=HTcount,as.matrix(D2%*%obj$v),stringsAsFactors=FALSE)
  uhtMat=rbind(uMat[,c(-1,-2)],htMat[,c(-1,-2)])
  row.names(uhtMat)=c(uMat$ID,htMat$ID)
  
  if (verbose){
    cat('spectral features extracted... clustering\n')
  }
  
  ht_kobj=kmeans(uhtMat,k,iter.max=10000,algorithm='Lloyd')
  uMat=data.frame(uMat[,1:2],topic_cluster=ht_kobj$cluster[1:dim(uMat)[1]])
  htMat=data.frame(htMat[,1:2],topic_cluster=ht_kobj$cluster[(dim(uMat)[1]+1):length(ht_kobj$cluster)])
  names(htMat)[1]='hashtag'
  summ=as.data.frame(ftable(uMat$topic_cluster))
  names(summ)=c('cluster','count')
  summ=summ[order(summ$count,decreasing=TRUE),]
  return(list(summary=summ,users=uMat,hashtags=htMat))
} 

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
# source(paste0(getwd(),"/util.R"))

# You can optionally identify some users (we do this via ID) and hashtags that you are particularly interested in here
# known_important_users <- c("29417304","755113","16948493")
# known_important_hashtags <- c("charlotteprotests", "keithlamontscott",
#                               "charlotte","charlotteriot","charlottepd",
#                               "charlotteuprising","blacklivesmatter")

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
# csv_output_name <- paste0("bsc_", MIN_USER, "_", N_CLUSTERS)
# write.csv(listObj$summary, file.path(OUTPUT_DIRECTORY, paste0(csv_output_name,'_summary.csv')))
# write.csv(listObj$users, file.path(OUTPUT_DIRECTORY, paste0(csv_output_name,'_users.csv')))
# write.csv(listObj$hashtags, file.path(OUTPUT_DIRECTORY, paste0(csv_output_name,'_hashtags.csv')))

# output_filename
# output_filename <- paste0("hashtags_per_cluster_",MIN_USER,"_",N_CLUSTERS,"_.pdf")

# look at results
# gen_plots(listObj,min_user_count = 1,filename=file.path(OUTPUT_DIRECTORY, output_filename))

#test loop
for(k in 10:500){
  for (l in 1:100){
    l <- l/100
    zzz <- matrix(0, ncol=3383, nrow=2363)
    diag(zzz) <- l
    
    obj=irlba(An+zzz,k,nu=k,nv=k,maxit = 2000, tol=1e-4, verbose=TRUE, work=2000)
  }
}

