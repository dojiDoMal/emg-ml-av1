// https://www.freecodecamp.org/news/the-least-squares-regression-method-explained/
// https://pt.wikipedia.org/wiki/M%C3%A9todo_dos_m%C3%ADnimos_quadrados

/* 
* perceptron simples | função H(.)
* y = H(x)
* 
* Erro - e(t) entre a saida desejada - d(t) e a saida gerada pela rede - y(t)
* e(t) = d(t) - y(t)
*
*/

// Fórmula da Porta lógica OR (Reta)
// X2 = -(W1/W2)*X1 + THETA/W2
// Para classes linearmente separáveis

function EMG(data){
    this.grumpy = data.Grumpy;
    this.surpreso = data.Surpreso;
    this.c = 2;
    this.p = 2;
}

// Permite visualizar uma cópia (screenshot) do objeto em dado momento
EMG.prototype.visualize = function EMG_visualize(){
    return JSON.parse(JSON.stringify(this));
}

function handleData(data /*, rodada*/){

    const seed = 45;

    let arAuxGr0 = [];
    let arAuxGr1 = [];
    let arAuxSu0 = [];
    let arAuxSu1 = [];

    for (const rod in data) {
        if (Object.hasOwnProperty.call(data, rod)) {
            arAuxGr0 = arAuxGr0.concat(data[rod].Grumpy[0])
            arAuxGr1 = arAuxGr1.concat(data[rod].Grumpy[1])   
            arAuxSu0 = arAuxSu0.concat(data[rod].Surpreso[0])
            arAuxSu1 = arAuxSu1.concat(data[rod].Surpreso[1])        
        }
    }
    
    const objD = {
        Grumpy: [arAuxGr0, arAuxGr1],
        Surpreso: [arAuxSu0, arAuxSu1]
    }

    //const dadosRodada = data[rodada];
    let emgObject = new EMG(objD);
    //console.log(dadosRodada)
    //console.log(objD)

    // Embaralha os valores
    //emgObject.grumpy[0] = shuffle(emgObject.grumpy[0], seed);
    //emgObject.grumpy[1] = shuffle(emgObject.grumpy[1], seed);

    //emgObject.surpreso[0] = shuffle(emgObject.surpreso[0], seed);
    //emgObject.surpreso[1] = shuffle(emgObject.surpreso[1], seed);

    // Cria os modelos de regressão linear
    /*
    const grumpyLR_model = calculateLinearRegression(emgObject.grumpy[0], emgObject.grumpy[1])
    const surpresoLR_model = calculateLinearRegression(emgObject.surpreso[0], emgObject.surpreso[1])
    
    // Cria as retas para plotar no gráfico
    const grumpyLine = {
        a: grumpyLR_model.a,
        b: grumpyLR_model.b,
        min: 0,
        max: 3000,
        class: 'Grumpy'
    }

    const surpresoLine = {
        a: surpresoLR_model.a,
        b: surpresoLR_model.b,
        min: 0,
        max: 3000,
        class: 'Surpreso'
    }
    */

    // Cria os tensores
    //const X_grumpy = tf.tensor2d([emgObject.grumpy[0], emgObject.grumpy[1]]);
    const Y_grumpy = tf.ones([1, emgObject.grumpy[0].length]);
    const grumpyFinal = tf.tensor2d([emgObject.grumpy[0], emgObject.grumpy[1], Y_grumpy.arraySync()]).transpose();
    //   X1    X2   Y
    // [1862, 1593, 1],
    // [1885, 1655, 1],
    // [1881, 1663, 1], ...
    grumpyFinal.print()

    //const X_surpreso = tf.tensor2d([emgObject.surpreso[0], emgObject.surpreso[1]]);
    const Y_surpreso = tf.zeros([1, emgObject.surpreso[0].length]);
    const surpresoFinal = tf.tensor2d([emgObject.surpreso[0], emgObject.surpreso[1], Y_surpreso.arraySync()]).transpose();
    //   X1    X2   Y
    // [592 , 529 , 0],
    // [573 , 510 , 0],
    // [554 , 496 , 0], ...
    surpresoFinal.print()
    
    /*
    let aux0 = grumpyFinal.transpose().arraySync()[0]
    let aux1 = grumpyFinal.transpose().arraySync()[1]
    let aux2 = grumpyFinal.transpose().arraySync()[2]
    const shuffledGrumpy = tf.tensor([aux0, aux1, aux2]).transpose()

    aux0 = surpresoFinal.transpose().arraySync()[0]
    aux1 = surpresoFinal.transpose().arraySync()[1]
    aux2 = surpresoFinal.transpose().arraySync()[2]
    const shuffledSurpreso = tf.tensor([aux0, aux1, aux2]).transpose()
    */

    //const surpresoSensor1 = tf.tensor2d(emgObject.surpreso[0], [emgObject.surpreso[0].length, 1], 'int32');
    //const surpresoSensor2 = tf.tensor2d(emgObject.surpreso[1], [emgObject.surpreso[1].length, 1], 'int32');

    //X_grumpy.print();
    //Y_grumpy.print();  
    //grumpyFinal.print()  
    //shuffledGrumpy.print()

    //X_surpreso.print();
    //Y_surpreso.print();
    //surpresoFinal.print()
    //shuffledSurpreso.print();

    const axis = 0;
    const finalTensor = grumpyFinal.concat(surpresoFinal, axis)
    finalTensor.print()

    
    aux0 = shuffle(finalTensor.transpose().arraySync()[0], seed)
    aux1 = shuffle(finalTensor.transpose().arraySync()[1], seed)
    aux2 = shuffle(finalTensor.transpose().arraySync()[2], seed)
    const shuffledFinal = tf.tensor([aux0, aux1, aux2]).transpose()
    //console.log(JSON.stringify(shuffledFinal.arraySync()));
    
    //Calculo logistico
    let X = tf.tensor([aux0, aux1]).transpose()
    let y = tf.tensor([aux2]).transpose()
    let m = aux2.length
    let n = 2 // Dois sensores
    
    //One step is missing, before implementing the cost function. 
    //The input matrix X needs to add an intercept term. 
    //Only that way the matrix operations work for the dimensions of theta and matrix X.
    //
    //3. Em sala foi realizada uma discussao sobre a adicao de um vetor 
    //coluna de 1s no inicio da matriz de dados X.
    X = tf.concat([tf.ones([m, 1]), X], 1);

    let theta = tf.zeros([n+1, 1]) // Array(n+1).fill().map(()=>[0]) // [[0], [0], [0]]
    let cost = costFunction(theta, X, y)

    theta.print()
    X.print()
    y.print()

    console.log('cost: ', cost);
    console.log('\n');
    //console.log(grumpyFinal.arraySync()[0][0])

    // Plota o gráfico
    scatterPlot(emgObject /*, {traceLine: true, lines: [grumpyLine, surpresoLine]}*/);
}

getJsonData = async (filePath) => {
    const response = await fetch(filePath)
    const data = await response.json();
    return data;
}

getJsonData('./emg.json').then((res) => handleData(res/*, EnumRodadas.R1*/));

//console.log(dados)

//console.log(JSON.stringify(dados))
/*
async function learnLinear(){
    
    const grumpyValues = [];
    for(sd in emgObject.data.Rodada1.Grumpy){
        for(sv in sd){
            console.log(sv)
        }
        //grumpyValues.push(sd)
    }
    const shape = [];
    
    const X = tf.tensor2d([1,5,6,-4,8,3], [6,1], 'int32');

    document.getElementById('output_field').innerText = X;
}
*/
//emgObject.learnLinear();
