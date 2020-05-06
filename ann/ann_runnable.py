import ann.ann_network, torch, numpy


def dataset_splitted_half(x_tensor, y_tensor, size):
    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    half_mark = int(len(dataset) * 0.5) 
    checksum = len(dataset) % 2 
    half_mark += checksum 

    train, test = torch.utils.data.random_split(dataset, [half_mark, len(dataset) - half_mark ])
    trainset = torch.utils.data.DataLoader(train, batch_size=size, shuffle=True)
    testset = torch.utils.data.DataLoader(test, batch_size=size, shuffle=True)
    return trainset, testset

def saveResults( szPath : str, lResults : list):
    with open(szPath, "w") as f:
        f.write("LOSS,LEARNING_RATE,MOMENTUM,NEURON_COUNT,HIT,TOTAL,EPOCH\n")
        for result in lResults:
            for element in result[:-1]:
                f.write( str(element)+",") 
            f.write( str(result[-1])+"\n" ) 

def train_network(numFeautes, trainSet, testSet, listNeurons, stepMomentum, epochCount):
    listResults = []
    Result = tuple() 

    ann.ann_network.depth += 1
    for currNeuron in listNeurons:
        ann.ann_network.LOG("--- Tworzenie sieci z liczba neuronow: " + str(currNeuron) +  " ---")
        curr_net =  ann.ann_network.Net(numFeautes, currNeuron )  
        listLearningRates = [0.001]
        ann.ann_network.depth += 1
        for currLearningRate in listLearningRates:
            currMomentum = 0
            while currMomentum < 1:
                ann.ann_network.LOG("--- Momentum: " + str(currMomentum) + ", Liczba neuronów: "+ str(currNeuron) + 
                        ", Stopień uczenia: " + str(currLearningRate) + " ---")
                currOptimizer = torch.optim.SGD( curr_net.parameters(), lr = currLearningRate, momentum=currMomentum )
                ann.ann_network.depth += 1
                for currEpoch in range(epochCount):
                    ann.ann_network.LOG("--- Epoch: " + str(currEpoch+1) + "/" + str(epochCount) + " ---")
                    loss = None
                    for currData in trainSet:
                        x, y = currData
                        curr_net.zero_grad()
                        result = curr_net(x)
                        loss = torch.nn.functional.nll_loss( result, y )
                        loss.backward()
                        currOptimizer.step()
                ann.ann_network.depth -= 1
                ann.ann_network.LOG("--- LOSS: " + str(loss.item()) + " ---")
                hit = 0
                total = 0
                with torch.no_grad():
                    for currData in testSet:
                        x, y = currData
                        output = curr_net( x ) 
                        for y_index,i in enumerate(output):
                            if torch.argmax( i ) == y[y_index]:
                                hit += 1
                            total += 1
                    perc = (hit*100)/total
                ann.ann_network.LOG("--- HIT/TOTAL RATIO: " + str(perc) + " ---" ) 

                
                Result = ( loss.item(), currLearningRate, currMomentum, currNeuron, hit, total, epochCount )
                listResults.append( Result )

                currMomentum += stepMomentum
        ann.ann_network.depth -= 1

    ann.ann_network.depth -= 1
    return listResults


def getAvgResults( llResults, passCount ):
    toRet = []

    count = len(llResults[0])
    for i in range(count - 1):
        loss = float(0)
        hit  = float(0)
        for j in range(passCount - 1):
            loss += llResults[j][i][0]
            hit += llResults[j][i][4]
        loss /= passCount
        hit /= passCount
        toRet.append( (loss, llResults[0][i][1], llResults[0][i][2], llResults[0][i][3], hit, llResults[0][i][5], llResults[0][i][6]) )

    return toRet

import ann.read_csv_function 
def start(szFilename, listNeurons, listFeatures, stepMomentum, batchSize):
    ann.ann_network.LOG(" --- Pobieranie danych ---")
    #x_csv, y_csv = ann.ann_network.csvToData()
    x_csv, y_csv = ann.read_csv_function.read_csv( szFilename )
    ann.ann_network.LOG(" --- Filtruj cechy ---")
    # ----------------------------------
    # Tworzenie neuronów
    ann.ann_network.LOG(" --- Tworzenie neuronów we/wy ---")
    x_tensor = torch.Tensor(x_csv.values)
    y_tensor = torch.from_numpy(y_csv.values)
    # ---------------------------------
    # Tworzenie datasetu podzielonego na pół
    ann.ann_network.LOG(" --- Dzielenie danych na pół ---")
    train, test = dataset_splitted_half(x_tensor, y_tensor, 4)
    # ---------------------------------
    # Trenowanie sieci
    ann.ann_network.LOG(" --- Trenowanie sieci na danych trenujących i testowanie na testujących ---")
    saveResults("output_normal.csv", train_network( len(x_csv.columns) ,train, test, listNeurons, stepMomentum, 10) )

