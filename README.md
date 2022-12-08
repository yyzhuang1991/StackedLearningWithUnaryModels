## Exploiting Unary Relations with Stacked Learning for Relation Extraction

 This repository contains the source codes of our paper: 
    [Exploiting Unary Relations with Stacked Learning for Relation Extraction](https://aclanthology.org/2022.sdp-1.14/)

---
### Set Up Environment
Our codes run with Python3 and PyTorch. Please install the required packages by the following command: 
    
    pip install -r requirements.txt

We also require the CoreNLP package for the data preprocessing. Please install StanfordCoreNLP-4.2.0 by following the instruction at [the site of CoreNLP](https://stanfordnlp.github.io/CoreNLP/) 

---
### Download Data 
Our experiments mainly use the LPSC dataset, which could be collected from [site1 (lpsc15 and lpsc16)](https://zenodo.org/record/1048419#.Y5FaYOzMIW8) and [site2 (mer-a, phx and mpf)](https://zenodo.org/record/7066107#.Y5ENJezMIW8). The dataset contains human annotations for the task of named entity recognition and relation extraction over planetary scientific publications relevant to several Mars missions. Please use the following command to collect the dataset: 

    bash get_lpsc_data.sh 
   
Note that we did not use the data of `mer-b` from [site2](https://zenodo.org/record/7066107#.Y5ENJezMIW8) in our paper.  

---
### Experiments on LPSC

In our paper, we performed a 5-fold cross validation experiment over the LPSC dataset. In the rest of this section, we show how to run experiments over LPSC step by step. 

#### 1. Data Preprocesseing

First, we parse the texts with CoreNLP, such as tokenizing and sentence segmentation. The following command runs the preprocessing script by giving it the absolute path to the stanford-corenlp directory: 

    bash preprocess.sh <ABSOLUTE/PATH/TO/stanford-corenlp-4.2.0>

We then split the files (train-dev-test) for 5-fold cross validation.     

    cd scripts
    python makeKFolds.py --outdir lpsc_5fold

Note that we first use about 25% files in the original dataset to construct a development set. Then we use the rest to create train-test data for the 5 fold cross validation. In other words, the development set stays the same for each run in the 5-fold cross validation experiments.        

#### 2. Train Unary and Binary Relation Extraction Models

Next, we train the unary and binary relation models. In the following instruction, we show how to run experiments for the relation `Contains`. The same things would apply to the relation `HasProperty`. 

First let's set the relation: 

    relation=Contains 

We also set the base encoder to `bert-base-uncased`, but you could try different types of encoders (see the options for `--modelType` in `UBmodel/run_clf.py`). 

    modelType=bert-base-uncased


##### 2.1. Make data objects

We convert files with annotation to data objects that we use to train unary and binary relation extraction models. The following commands create data objects for the relation `Contains`: 

    cd UBmodel/ 

    dataSplitDir=../lpsc_5fold # directory that stores the data splits
    outdir=lpsc_instances_5fold # output directory 
    mode='both'
    mkdir $outdir
    
    for fold in {0..4}
    do
        python convert_data.py \
        --mode 'both' \
        --relation $relation \
        --trainList "$dataSplitDir"/fold"$fold"/train.list \
        --devList "$dataSplitDir"/fold"$fold"/dev.list \
        --testList "$dataSplitDir"/fold"$fold"/test.list \
        --outdir "$outdir"/fold"$fold" 
    done 

Arguments:
   * mode: specifies what type of relation extraction models to produce data objects for. This could be 'unary', 'binary' or both. 
   * trainList: a file that contains a list of files for training 
   * devList: a file that contains a list of files for development 
   * testList: a file that contains a list of files for testing
   * outdir: output directory to save the data objects to 
 

##### 2.2. Train Unary Models 

Next, we train two unary models to extract the unary relations `Contains(*, Component)` and `Contains(Target, *)` separately. The following command trains both models one by one over the `fold0` data.

    fold=0
    epoch=4
    lr=1e-5
    dataDir=lpsc_instances_5fold/fold"$fold"
    
    # ------- train Contains(Target, *)
    ner=Target 
    indirU1="$dataDir"/"$ner"-"$relation" # directory where the input data objects are for this particular unary relation
    umodel1=output/lr"$lr"_epoch"$epoch"/fold"$fold"/"$ner"-"$relation" # directory to save the model
    mkdir -p $umodel1
    python run_clf.py \
        --mode unary \
        --train \
        --modelType $modelType \
        --trainDir "$indirU1"/train \
        --testDir "$indirU1"/dev  \
        --lr $lr \
        --epoch $epoch \
        --modelSaveDir $umodel1
    
    # ------- train Contains(*, Component)
    ner=Component 
    epoch=4
    lr=1e-5
    indirU2="$dataDir"/"$ner"-"$relation"
    umodel2=output/lr"$lr"_epoch"$epoch"/fold"$fold"/"$ner"-"$relation"
    mkdir -p $umodel2
    python run_clf.py \
        --mode unary \
        --train \
        --modelType $modelType \
        --trainDir "$indirU2"/train \
        --testDir "$indirU2"/dev  \
        --lr $lr \
        --epoch $epoch \
        --modelSaveDir $umodel2

##### 2.3. Train Binary Relation Extraction Model
Next we train a binary relation extraction model that extracts `Contains(Target, Component)`by the following command: 
    
    if [ $relation = "Contains" ]
    then  
        binaryRelation=Component-Target-Contains
    else
        binaryRelation=Property-Target-HasProperty 
    fi 
    
    lr=2e-5
    epoch=10
    indirB="$dataDir"/"$binaryRelation" # directory where the data objects are stored
    bmodel=output/lr"$lr"_epoch"$epoch"/fold"$fold"/"$binaryRelation" # directory where the model is saved
    mkdir -p $bmodel

    python3 run_clf.py \
        --mode binary \
        --modelType $modelType \
        --train \
        --trainDir "$indirB"/train  \
        --testDir "$indirB"/dev  \
        --lr $lr \
        --epoch $epoch \
        --modelSaveDir $bmodel

##### 2.4. Make Predictions Using The Unary And Binary Models

Next we use the unary and binary relation extraction models to make predictions over the training, development and testing sets. This step is necessary since the meta-classifier needs the outputs of these three models to train and make inference. The predictions would be saved to the directory specified by `evalOutdir`. 
    
    # predict with the unary models
    for testName in train test dev
    do 
        for umodel in $umodel1 $umodel2
        do
                
                if [ $umodel = $umodel1 ]
                then
                    indir=$indirU1
                else
                    indir=$indirU2
                fi 

                evalOutdir="$umodel"/"$testName" #output directory to save predictions
                mkdir $evalOutdir
                python run_clf.py \
                --mode unary \
                --test \
                --testDir "$indir"/"$testName" \
                --trainedModelDir $umodel \
                --evalOutdir $evalOutdir 
        done
    done

    #predict with the binary model
    for testName in train test dev
    do 
        evalOutdir="$bmodel"/"$testName" #output directory to save predictions
        python run_clf.py \
            --mode binary \
            --test \
            --testDir "$indirB"/"$testName" \
            --trainedModelDir $bmodel \
            --evalOutdir $evalOutdir
    done

At this point, we have extracted unary relation instances and binary relation instances using the unary and binary models. Do the following command to return to the upper directory:
    
    cd ..

#### 3. Train The Meta classifier
To train the meta classifier, we move to the `meta_extraction/` folder first:
    
    cd meta_extraction 

##### 3.1. Convert Data 
First convert the data for the meta classifier, so the data objects for the meta classifier contains the predictions from the unary and the binary models.

    umodel1=../UBmodel/"$umodel1" # directory where the predictions of the first unary model were saved
    umodel2=../UBmodel/"$umodel2" # directory where the predictions of the 2nd unary model were saved
    bmodel=../UBmodel/"$bmodel" # directory where the predictions of the binary model were saved
    metaDataDir=data # output directory to save the data objects
    
    python convert_data.py \
        --relation $relation \
        --trainList "$dataSplitDir"/fold"$fold"/train.list \
        --devList "$dataSplitDir"/fold"$fold"/dev.list \
        --testList "$dataSplitDir"/fold"$fold"/test.list \
        --trainU1Predictions "$umodel1"/train/predInstances.pkl \
        --devU1Predictions "$umodel1"/dev/predInstances.pkl \
        --testU1Predictions "$umodel1"/test/predInstances.pkl \
        --trainU2Predictions "$umodel2"/train/predInstances.pkl \
        --devU2Predictions "$umodel2"/dev/predInstances.pkl \
        --testU2Predictions "$umodel2"/test/predInstances.pkl \
        --trainBinaryPredictions "$bmodel"/train/predInstances.pkl \
        --devBinaryPredictions "$bmodel"/dev/predInstances.pkl \
        --testBinaryPredictions "$bmodel"/test/predInstances.pkl \
        --outdir $metaDataDir

##### 3.2. Train And Test A Meta Classifier

Next we train a meta classifier by the following command: 
    
    outDir=output
    python run_skmodel.py \
        --train \
        --test \
        --trainDir "$metaDataDir"/"$binaryRelation"/train \
        --testDir "$metaDataDir"/"$binaryRelation"/dev \
        --modelOutdir $outDir

Finally, we test the meta classifier over the development set and the test set: 

    for name in dev test
    do
        
        python run_skmodel.py \
            --test \
            --testDir "$outDir"/"$binaryRelation"/"$name" \
            --trainedModelDir $outDir
    done

which prints out the evaluation score. 

---
### Citation

Please cite our work if you use it in your research: 
    
    @inproceedings{zhuang-etal-2022-exploiting,
        title = "Exploiting Unary Relations with Stacked Learning for Relation Extraction",
        author = "Zhuang, Yuan and Riloff, Ellen and Wagstaff, Kiri L. and Francis, Raymond  and Golombek, Matthew P. and Tamppari, Leslie K.",
        booktitle = "Proceedings of the Third Workshop on Scholarly Document Processing",
    }

Please also cite the following works if you use the LPSC dataset in your research: 
   
   + Kiri L. Wagstaff, Raymond Francis, Thamme Gowda, You Lu, Ellen Riloff, Karanjeet Singh, and Nina Lanza. "Mars Target Encyclopedia: Rock and Soil Composition Extracted from the Literature."  Proceedings of the Thirtieth Annual Conference on Innovative Applications of Artificial Intelligence, 2018.


   + Kiri L. Wagstaff, Raymond Francis, Matthew Golombek, Leslie Tamppari, and Steven Lu. (2022). Mars Target Encyclopedia - Labeled LPSC abstracts for four Mars missions (1.0.0.0) [Data set]. Zenodo. DOI: 10.5281/zenodo.7066107




