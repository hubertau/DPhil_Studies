# attempt at bipartite graph viz

# see https://stackoverflow.com/questions/60100006/visualize-bipartite-network-graph-created-using-pandas-dataframe

mapping=!grepl("^[0-9]+$", V(H)$name)
V(H)$type <- mapping
l <- layout.bipartite(H)
plot(H, layout = l[, c(2,1)])
