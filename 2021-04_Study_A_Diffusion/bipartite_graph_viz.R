# attempt at bipartite graph viz

# see https://stackoverflow.com/questions/60100006/visualize-bipartite-network-graph-created-using-pandas-dataframe

files <- list.files(path='/Users/hubert/Nextcloud/DPhil/DPhil_Studies/2021-04_Study_A_Diffusion/collection_results_2021_05_04_16_22/bsc/',
                    pattern="*.RData$",
                    full.names = TRUE)

# see which ones have convergence or clustering errors
irl_warns <- c()
irl_errors <- c()
kmeans_warns <- c()
kmeans_errors <- c()
cluster_sizes <- c()
for (file in files){
  # read in test data
  load(file)
  cluster_sizes <- c(cluster_sizes, max(listObj$ht_kobj$cluster))
  irl_warns     <- c(irl_warns, listObj$irl_warn)
  irl_errors     <- c(irl_errors, listObj$irl_error)
  kmeans_warns     <- c(kmeans_warns, listObj$kmeans_warnings)
  kmeans_errors     <- c(kmeans_errors, listObj$kmeans_errors)
  
}


# mapping=!grepl("^[0-9]+$", V(H)$name)
# V(H)$type <- mapping
# l <- layout.bipartite(H)
# H$x <- l[,2]
# H$y <- l[,1]

# E(H)$color <- 'gray'
# E(g)$color[E(g)['A' %--% V(g)]] <- 'red'
# E(g)$color[E(g)['B' %--% V(g)]] <- 'blue'
# E(g)$color[E(g)['C' %--% V(g)]] <- 'green'

plot(listObj$adjacency,
    
          
)