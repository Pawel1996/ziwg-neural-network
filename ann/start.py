def start(listNeurons, listFeatures, stepMomentum, batchSize):
    ann.ann_network.LOG(" --- Pobieranie danych ---")
    # x_csv, y_csv = ann.ann_network.csvToData()
    x_csv, y_csv = read_csv("../ZWIG-Bigramy/correlation-8.csv")

    ann.ann_network.LOG(" --- Filtruj cechy ---")
    # ----------------------------------
    # Tworzenie neuronów
    ann.ann_network.LOG(" --- Tworzenie neuronów we/wy ---")
    x_tensor = torch.Tensor(x_csv.values)
    y_tensor = torch.Tensor(y_csv.values)
    # y_tensor = torch.from_numpy(y_csv)

    # ---------------------------------
    # Tworzenie datasetu podzielonego na pół
    ann.ann_network.LOG(" --- Dzielenie danych na pół ---")
    train, test = dataset_splitted_half(x_tensor, y_tensor, 4)

    # ---------------------------------
    # Trenowanie sieci
    ann.ann_network.LOG(" --- Trenowanie sieci na danych trenujących i testowanie na testujących ---")
    saveResults("output_normal.csv", train_network(train, test, listNeurons, stepMomentum, 10))
