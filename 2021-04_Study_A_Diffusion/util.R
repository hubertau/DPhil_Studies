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
load_user_ht_matrix <- function(edgelist_path){
  cat('constructing graph')
  h_edges=fread(edgelist_path,stringsAsFactors=FALSE,colClasses = c("character","character","integer"))
  setnames(h_edges, c("Source","Target","weight"))
  h_edges$Target=paste('#',h_edges$Target,sep='')
  h_edges <- data.table(h_edges)
  
}

biSpectralCoCluster=function(h_edges,min_user=1,k=100,all_hashtags=FALSE,verbose = FALSE){
  
  H=graph.data.frame(h_edges)
  S=simplify(H,remove.loops=FALSE,remove.multiple=TRUE)
  rm(h_edges)
  A=get.adjacency(H,names=TRUE)
  S=get.adjacency(S)
  
  # HUBERT NOTE: THIS NEXT LINE TREATS ENTRIES IN TARGET COLUMN AS HASHTAGS
  if (all_hashtags){
    mapping=grepl('#',V(H)$name)  
  }else{
    mapping=!grepl("^[0-9]+$", V(H)$name)
  }
  
  A=A[,mapping]
  S=S[,mapping]
  rm(H)
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
  obj=irlba(An,k,nu=k,nv=k)
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


gen_plots <- function(listObj,min_user_count=50, filename='cluster_res_new.pdf', clusters_of_interest=NA){
  userData = listObj[["users"]]
  htData <- listObj[["hashtags"]]
  summ <- listObj[["summary"]]
  top_ht <- 25
  
  
  cairo_pdf(paste(filename,sep=''),width=7,height=10)
  # quartz(type = 'pdf', file = paste(filename,sep=''),width=7,height=10)
  
  clusters_to_run <- 1:max(userData$topic_cluster)
  if(!is.na(clusters_of_interest)){
    clusters_to_run = clusters_of_interest
    min_user_count = 0
  }
  
  for(cluster in clusters_to_run){
    #pdf(paste('img/cluster_',cluster,'.pdf',sep=''),width=7,height=10)
    uD=userData[userData$topic_cluster == cluster,]
    htD=htData[htData$topic_cluster == cluster,]
    if(nrow(uD)< min_user_count | nrow(htD) == 0){
      next
    }
    ht_c <- min(top_ht,nrow(htD))
    uD=uD[order(uD$degree,decreasing=TRUE),]
    htD=htD[order(htD$degree,decreasing=TRUE),]
    # 2021-06-16 Hubert Note: do not make the hashtags 
    # htD$hashtag=gsub('#','',htD$hashtag)
    
    p1 <- ggplot(arrange(htD, -degree)[1:ht_c,], aes(reorder(hashtag,degree),degree)) + geom_bar(stat='identity') + coord_flip()
    p1 <- p1 + ylab("Times Used") + xlab("Hashtag")
    suppressWarnings(print(p1 + annotate("text",x=4,y=max(htD$degree,na.rm = T)-.35*max(htD$degree), label=paste("Cluster: ",cluster," n users:",nrow(uD)))))
    #dev.off()
  }
  garbage <- dev.off()
}
