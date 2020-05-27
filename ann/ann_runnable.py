import ann.ann_network, torch, numpy
import ann.read_csv_function
import re

def dataset_splitted_half(x_tensor, y_tensor, size):
    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    split_mark = int(len(dataset) * 0.7) 
    checksum = len(dataset) % 7
    split_mark += checksum 

    train, test = torch.utils.data.random_split(dataset, [split_mark, len(dataset) - split_mark ])
    trainset = torch.utils.data.DataLoader(train, batch_size=size, shuffle=True)
    testset = torch.utils.data.DataLoader(test, batch_size=size, shuffle=True)
    return trainset, testset

def clearClassList(classList):
    for elem in classList:
        elem[1] = 0
        elem[2] = 0
    return


def train_network(numFeatures, trainSet, testSet, listNeurons, stepMomentum, epochList, listLayers, clList):
    listResults = []
    Result = tuple() 
    ann.ann_network.depth += 1
    for currLayers in listLayers:
        for currNeuron in listNeurons:
            for currEpoch in epochList:
                currMomentum = 0
                while currMomentum <= 0.98:
                    clearClassList(clList)
                    ann.ann_network.LOG("--- Tworzenie sieci z liczba neuronow: " + str(currNeuron) +  
                             "Liczba warstw: " + str(currLayers) + " Z liczbą epok: "+  str(currEpoch) + " ---")
                    curr_net =  ann.ann_network.Net(numFeatures, currNeuron, currLayers)
                    # Testowanie liczby warstw
                    # for name, param in curr_net.named_parameters():
                    # if param.requires_grad:
                    # print(name, param.data)
                    ann.ann_network.depth += 1
                    ann.ann_network.LOG("--- Momentum: " + str(currMomentum) + ", Liczba neuronów: "+ str(currNeuron) + 
                        ", Stopień uczenia: 0.001  ---")
                    currOptimizer = torch.optim.SGD( curr_net.parameters(), lr = 0.001, momentum=currMomentum )
                    ann.ann_network.depth += 1
                    for indexEpoch in range(currEpoch):
                        ann.ann_network.LOG("--- Epoch: " + str(indexEpoch+1) + "/" + str(currEpoch) + " ---")
                        loss = None
                        for currData in trainSet:
                            x, y = currData
                            curr_net.zero_grad()
                            result = curr_net(x)
                            loss = torch.nn.functional.nll_loss( result, y.long() )
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
                                    clList[ y[y_index] ][2] += 1
                                    hit += 1
                                total += 1
                                clList[ y[y_index] ][1] += 1
                    perc = (hit*100)/total 
                    ann.ann_network.LOG("--- HIT/TOTAL RATIO: " + str(perc) + " ---" )

                    Result = [ loss.item(), 0.001, currLayers, currMomentum, currNeuron, hit, total, currEpoch, numFeatures ]
                    for elem in clList:
                        Result = Result + [ elem[1], elem[2] ]
                    listResults.append( Result )
                    currMomentum += stepMomentum
                    ann.ann_network.depth -= 1

    ann.ann_network.depth -= 1
    return listResults

def saveResults( szPath : str, lResults : list, classList):
    with open(szPath, "w") as f:
        f.write("LOSS,LEARNING_RATE,LAYER_COUNT,MOMENTUM,NEURON_COUNT,HIT,TOTAL,EPOCH,FEATURE_COUNT,")
        for elem in classList[:-1]:
            f.write(elem[0]+'_totals,')
            f.write(elem[0]+'_hits,')

        f.write(classList[-1][0]+'_totals,')
        f.write(classList[-1][0]+'_hit\n')

        for result in lResults:
            for element in result[:-1]:
                f.write( str(element)+",") 
            f.write( str(result[-1])+"\n" )
def getClassList(szClassMeta):
    toRet = []
    with open(szClassMeta, "r") as f:
        while True:
            line = f.readline()
            if line == "":
                break

            clLine = re.split('\\|' , line)
            clLine[0] = clLine[0].replace(' ', '_')
            toRet.append( [clLine[0], 0, 0] )

    return toRet
def start(szFilename, szDest, listNeurons, stepMomentum,layersList, epochList):
    ann.ann_network.LOG(" --- Pobieranie danych ---")
    clList = getClassList("class.meta")
    x_csv, y_csv = ann.read_csv_function.read_csv( szFilename )
    # ----------------------------------
    # Tworzenie neuronów
    ann.ann_network.LOG(" --- Tworzenie neuronów we/wy ---")
    x_tensor = torch.Tensor(x_csv.values)
    y_tensor = torch.Tensor(y_csv).long()
    print(x_tensor)
    print(y_tensor)
    # ---------------------------------
    # Tworzenie datasetu podzielonego na pół
    ann.ann_network.LOG(" --- Dzielenie danych na pół ---")
    train, test = dataset_splitted_half(x_tensor, y_tensor, 4)
    # ---------------------------------
    # Trenowanie sieci
    ann.ann_network.LOG(" --- Trenowanie sieci na danych trenujących i testowanie na testujących ---")
    saveResults(szDest, train_network( len(x_csv.columns) ,train, test, listNeurons, stepMomentum, epochList, layersList, clList), clList )

