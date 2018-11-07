algaes = read.table(file=file.choose(), 
                    header=T, 
                    sep=","
                    )
dataSet = algaes[,c("season")]


barplot(prop.table(table(dataSet)))


