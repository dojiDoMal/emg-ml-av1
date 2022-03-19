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

function linearRegression(data){

    let seed = 25;

    let arAuxGr0 = [];
    let arAuxGr1 = [];
    let arAuxSu0 = [];
    let arAuxSu1 = [];

    for (const rod in data) {
        if(rod == "Rodada5") break;
        if (Object.hasOwnProperty.call(data, rod)) {
            arAuxGr0 = arAuxGr0.concat(data[rod].Grumpy[0])
            arAuxGr1 = arAuxGr1.concat(data[rod].Grumpy[1])   
            arAuxSu0 = arAuxSu0.concat(data[rod].Surpreso[0])
            arAuxSu1 = arAuxSu1.concat(data[rod].Surpreso[1])        
        }
    }
    
    const objData = {
        Grumpy: [arAuxGr0, arAuxGr1],
        Surpreso: [arAuxSu0, arAuxSu1]
    }

    let emgObject = new EMG(objData);
    
    // Cria os tensores
    const Y_grumpy = tf.ones([1, emgObject.grumpy[0].length*2]);
    const grumpyFinal = tf.tensor2d([emgObject.grumpy[0], emgObject.grumpy[1], Y_grumpy.arraySync()]).transpose();
    //   X1    X2   Y
    // [1862, 1593, 1],
    // [1885, 1655, 1],
    // [1881, 1663, 1], ...

    const Y_surpreso = tf.zeros([1, emgObject.surpreso[0].length*2]);
    const surpresoFinal = tf.tensor2d([emgObject.surpreso[0], emgObject.surpreso[1], Y_surpreso.arraySync()]).transpose();
    //   X1    X2   Y
    // [592 , 529 , 0],
    // [573 , 510 , 0],
    // [554 , 496 , 0], ...
    
    // Dados são concatenados verticalmente
    const axis = 0;
    const finalTensor = grumpyFinal.concat(surpresoFinal, axis)
    const tensorFX1 = finalTensor.transpose().arraySync()[0];
    const tensorFX2 = finalTensor.transpose().arraySync()[1];
    const tensorFY = finalTensor.transpose().arraySync()[2];

    const treino80 = tensorFX1.length*.8
    const teste20 = tensorFX1.length*.2
  
    let rodadas = 5
    let acuracia = []
    let sensibilidade = []
    let especificidade = []
    
    for (let i = 0; i < rodadas; i++) {
            
        // Dados sao embaralhados
        let aux0 = shuffle( tensorFX1, seed )
        let aux1 = shuffle( tensorFX2, seed )
        let auxY = shuffle( tensorFY, seed )
        seed++;    
        // 1) a. Faca a implementacao do metodo dos minimos quadrados ordinario
        const X = tf.tensor([aux0, aux1]).transpose()
        //const y = tf.tensor([auxY]).transpose()
        //let n = 2 // Dois sensores
        
        // Divide 80% para treino e 20% para teste
        const Xtreino = tf.tensor( [aux0.slice(0,treino80), aux1.slice(0,treino80)] ).transpose()
        //Xtreino.print(true)
        const yTreino = tf.tensor( [auxY.slice(0,treino80)] ).transpose()
        //yTreino.print(true)
        const Xteste = tf.tensor( [aux0.slice(-teste20), aux1.slice(-teste20)] ).transpose()
        //Xteste.print(true)
        const yTeste = tf.tensor( [auxY.slice(-teste20)] ).transpose()
        //yTeste.print(true)

        // Calculo de W = (Xt*X)^-1 * XtY 
        // Xt
        const Xt = Xtreino.transpose()
        // (Xt*X)^-1
        const XtXinv = tf.tensor(
            math.inv(
                tf.matMul( Xt, Xtreino ).arraySync() 
            ) 
        )        
        // XtY
        const XtY = tf.matMul(Xt, yTreino)
        const W = tf.matMul(XtXinv, XtY)

        //document.getElementById('plotResult').innerText = tf.matMul(Xt, Xtreino)
        
        emgObject.params = {
            w1: W.arraySync()[0],
            w2: W.arraySync()[1],
            theta: 0.5 
        }

        //ynovo = tf.matMul(Xteste, W)
        //ynovo.print()

        // Matriz de confusao
        let VP = 0
        let VN = 0
        let FP = 0
        let FN = 0

        for (let i = 0; i < Xteste.shape[0]; i++) {
            let previsao = tf.matMul( tf.tensor([Xteste.arraySync()[i]]), W ).arraySync()[0]
            let real = yTeste.arraySync()[i]

            // Aplica o degrau unitario nos valores obtidos 
            // com o modelo e gera a matriz de confusao
            if (unitStep(previsao) == real){
                (real == 0 ? VN++ : VP++)
            } else {
                (real == 0 ? FP++ : FN++)
            }        
        }
        
        const matrizConfusao = tf.tensor( [[VP, FP],[FN, VN]] )
        matrizConfusao.print()

        acuracia.push( (VP+VN) / (VP+VN+FP+FN) )
        sensibilidade.push( (VP) / (VP+FN) )
        especificidade.push( (VN) / (VN+FP) )
    }

    console.log(acuracia)
    console.log(sensibilidade)
    console.log(especificidade)
    
    //One step is missing, before implementing the cost function. 
    //The input matrix X needs to add an intercept term. 
    //Only that way the matrix operations work for the dimensions of theta and matrix X.
    //
    //3. Em sala foi realizada uma discussao sobre a adicao de um vetor 
    //coluna de 1s no inicio da matriz de dados X.
    //X = tf.concat([tf.ones([m, 1]), X], 1);

    // Plota o gráfico
    scatterPlot(emgObject);
}

getJsonData = async (filePath) => {
    const response = await fetch(filePath)
    const data = await response.json();
    return data;
}

getJsonData('./emg.json').then((res) => linearRegression(res));

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
