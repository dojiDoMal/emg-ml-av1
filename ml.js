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

function simplePerceptron(data){

    let seed = 0;

    let arAuxGr0 = [];
    let arAuxGr1 = [];
    let arAuxSu0 = [];
    let arAuxSu1 = [];

    for (const rod in data) {
        if(rod == "Rodada2") break;
        if (Object.hasOwnProperty.call(data, rod)) {
            arAuxGr0 = arAuxGr0.concat(data[rod].Grumpy[0])
            arAuxGr1 = arAuxGr1.concat(data[rod].Grumpy[1])   
            arAuxSu0 = arAuxSu0.concat(data[rod].Surpreso[0])
            arAuxSu1 = arAuxSu1.concat(data[rod].Surpreso[1])        
        }
    }

    // normalização dos dados
    const arAuxX = normalize( arAuxGr0.concat(arAuxSu0) )
    arAuxGr0 = arAuxX.slice(0, arAuxGr0.length ) 
    arAuxSu0 = arAuxX.slice(arAuxSu0.length, arAuxX.length)

    const arAuxY = normalize( arAuxGr1.concat(arAuxSu1) )
    arAuxGr1 = arAuxY.slice(0, arAuxGr1.length ) 
    arAuxSu1 = arAuxY.slice(arAuxSu1.length, arAuxY.length)
        
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

    // Garbage collection
    Y_grumpy.dispose();

    const Y_surpreso = tf.mul(tf.ones([1, emgObject.surpreso[0].length*2]), -1);
    const surpresoFinal = tf.tensor2d([emgObject.surpreso[0], emgObject.surpreso[1], Y_surpreso.arraySync()]).transpose();
    //   X1    X2   Y
    // [592 , 529 , -1],
    // [573 , 510 , -1],
    // [554 , 496 , -1], ...
    
    // Garbage collection
    Y_surpreso.dispose();
    
    
    // Dados são concatenados verticalmente
    const axis = 0;
    const finalTensor = grumpyFinal.concat(surpresoFinal, axis)
    finalTensor.print(true)

    // Garbage collection
    surpresoFinal.dispose();
    grumpyFinal.dispose();

    let tensorFX1 = finalTensor.transpose().arraySync()[0] ;
    let tensorFX2 = finalTensor.transpose().arraySync()[1] ;
    let tensorFY = finalTensor.transpose().arraySync()[2] ;

    // Garbage collection
    finalTensor.dispose();

    const treino80 = tensorFX1.length*.8
    const teste20 = tensorFX1.length*.2

    const learningRate = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.9,1]
  
    let rodadas = 1
    let acuracia = []
    let sensibilidade = []
    let especificidade = []
    
    for (let i = 0; i < rodadas; i++) {
            
        // Dados sao embaralhados
        let aux0 = shuffle( tensorFX1, seed )
        let aux1 = shuffle( tensorFX2, seed )
        let auxY = shuffle( tensorFY, seed )
        seed++;  
        
        // Divide 80% para treino e 20% para teste
        const Xtreino = tf.tensor( [aux0.slice(0,treino80), aux1.slice(0,treino80)] ).transpose()
        const Xf = tf.concat([tf.mul(tf.ones([Xtreino.shape[0], 1]), -1), Xtreino], 1).transpose() 
        //Xf.print(true)
        //Xtreino.print(true)

        const yTreino = tf.tensor( [auxY.slice(0,treino80)] )
        //yTreino.print(true)

        const Xteste = tf.tensor( [aux0.slice(-teste20), aux1.slice(-teste20)] ).transpose()
        const Xt = tf.concat([tf.ones([Xteste.shape[0], 1]), Xteste], 1)
        //Xt.print(true)

        const yTeste = tf.tensor( [auxY.slice(-teste20)] )
        //yTeste.print(true)

        let Wp = [[0],[0],[0]]
        let erro = true;
        let epochs = 0

        while(erro){
            erro = false;
            for(let i = 0; i < Xf.shape[1]; i++){
                let xAmostra = [[Xf.arraySync()[0][i]], [Xf.arraySync()[1][i]], [Xf.arraySync()[2][i]]];
                let norm = math.norm(math.transpose(xAmostra)[0]);
                let xAmostraNorm = math.multiply(
                    xAmostra,
                    1/norm
                )
                //console.log(xAmostraNorm)
                //console.log(xAmostra)
                //Wp.transpose().print(true)
                let u = tf.dot(math.transpose(Wp), xAmostraNorm).arraySync()[0]
                let y = signal(u)
                Wp = math.add(
                    Wp,
                    math.multiply(
                        learningRate[1] * (yTreino.arraySync()[0][i] - y), xAmostraNorm
                    )
                )
                if(yTreino.arraySync()[0][i] != y){
                    erro = true;
                } 
                
            }  
            epochs++;
            console.log( "epoca: " + epochs + " | " + Wp[0][0].toFixed(2)+", "+Wp[1][0].toFixed(2)+", "+Wp[2][0].toFixed(2))
            //if(epochs == 20) break;
        }

        emgObject.params = {
            w1: Wp[1],
            w2: Wp[2],
            theta: Wp[0] 
        }

        // Plota o gráfico
        scatterPlot(emgObject); 
        
        // Garbage collection
        Xtreino.dispose();
        yTreino.dispose();

        // Matriz de confusao
        let VP = 0
        let VN = 0
        let FP = 0
        let FN = 0
      
        for (let i = 0; i < Xt.shape[0]; i++) {

            // TODO: tentar normalizar o vetor Xt e ver se da certo
            let previsao = tf.dot( Xt.arraySync()[i], Wp ).arraySync()[0]  
            let real = yTeste.arraySync()[0][i] 
            //console.log(previsao)
            // Aplica o degrau unitario nos valores obtidos 
            // com o modelo e gera a matriz de confusao
            if (signal(previsao) == real){
                (real == -1 ? VN++ : VP++)
            } else {
                (real == -1 ? FP++ : FN++)
            }    
            
            // Garbage collection
            previsao = null
            real = null;
        }

        // Garbage collection
        Xteste.dispose();
        Xt.dispose();
        yTeste.dispose();
        
        //const matrizConfusao = tf.tensor( [[VP, FP],[FN, VN]] )
        //matrizConfusao.print()

        acuracia.push( (VP+VN) / (VP+VN+FP+FN) )
        sensibilidade.push( (VP) / (VP+FN) )
        especificidade.push( (VN) / (VN+FP) )
    }

    //console.log(acuracia)
    //console.log(sensibilidade)
    //console.log(especificidade)

    // Garbage Collection
    tensorFX1 = null;
    tensorFX2 = null;
    tensorFY = null;
    
    // Gera arquivo CSV com os dados estatisticos:
    // Acuracia, sensibilidade e especificidade
    let Data = [acuracia, sensibilidade, especificidade]
    let length = acuracia.length
    //geraCSVcomDownload(Data, length)

    // Garbage Collection
    acuracia = null;
    sensibilidade = null;
    especificidade = null;
    emgObject = null;

}

function lambdaLinearRegressionOnes(data){

    let seed = 0;

    let arAuxGr0 = [];
    let arAuxGr1 = [];
    let arAuxSu0 = [];
    let arAuxSu1 = [];

    for (const rod in data) {
        if(rod == "Rodada6") break;
        if (Object.hasOwnProperty.call(data, rod)) {
            arAuxGr0 = arAuxGr0.concat(data[rod].Grumpy[0])
            arAuxGr1 = arAuxGr1.concat(data[rod].Grumpy[1])   
            arAuxSu0 = arAuxSu0.concat(data[rod].Surpreso[0])
            arAuxSu1 = arAuxSu1.concat(data[rod].Surpreso[1])        
        }
    }

    // normalização dos dados
    const arAuxX = normalize( arAuxGr0.concat(arAuxSu0) )
    arAuxGr0 = arAuxX.slice(0, arAuxGr0.length ) 
    arAuxSu0 = arAuxX.slice(arAuxSu0.length, arAuxX.length)

    const arAuxY = normalize( arAuxGr1.concat(arAuxSu1) )
    arAuxGr1 = arAuxY.slice(0, arAuxGr1.length ) 
    arAuxSu1 = arAuxY.slice(arAuxSu1.length, arAuxY.length)
    
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

    // Garbage collection
    Y_grumpy.dispose();

    const Y_surpreso = tf.zeros([1, emgObject.surpreso[0].length*2]);
    const surpresoFinal = tf.tensor2d([emgObject.surpreso[0], emgObject.surpreso[1], Y_surpreso.arraySync()]).transpose();
    //   X1    X2   Y
    // [592 , 529 , 0],
    // [573 , 510 , 0],
    // [554 , 496 , 0], ...
    
    // Garbage collection
    Y_surpreso.dispose();
    
    // Dados são concatenados verticalmente
    const axis = 0;
    const finalTensor = grumpyFinal.concat(surpresoFinal, axis)

    // Garbage collection
    surpresoFinal.dispose();
    grumpyFinal.dispose();

    let tensorFX1 = normalize( finalTensor.transpose().arraySync()[0] );
    let tensorFX2 = normalize( finalTensor.transpose().arraySync()[1] );
    let tensorFY = finalTensor.transpose().arraySync()[2];

    // Garbage collection
    finalTensor.dispose();

    const treino80 = tensorFX1.length*.8
    const teste20 = tensorFX1.length*.2
  
    let rodadas = 100
    let acuracia = []
    let sensibilidade = []
    let especificidade = []
    
    for (let i = 0; i < rodadas; i++) {
            
        // Dados sao embaralhados
        let aux0 = shuffle( tensorFX1, seed )
        let aux1 = shuffle( tensorFX2, seed )
        let auxY = shuffle( tensorFY, seed )
        seed++;  
        
        // Divide 80% para treino e 20% para teste
        const Xtreino = tf.tensor( [aux0.slice(0,treino80), aux1.slice(0,treino80)] ).transpose()
        const XtreinoUns = tf.concat([tf.ones([Xtreino.shape[0], 1]), Xtreino], 1)
        XtreinoUns.print(true)
        //Xtreino.print(true)
        const yTreino = tf.tensor( [auxY.slice(0,treino80)] ).transpose()
        //yTreino.print(true)
        const Xteste = tf.tensor( [aux0.slice(-teste20), aux1.slice(-teste20)] ).transpose()
        const XtesteUns = tf.concat([tf.ones([Xteste.shape[0], 1]), Xteste], 1)
        XtesteUns.print(true)
        //Xteste.print(true)
        const yTeste = tf.tensor( [auxY.slice(-teste20)] ).transpose()
        //yTeste.print(true)

        // Calculo de W = (Xt*X + λI)^-1 * XtY 
        // λ
        const lambda = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.9,1]
        // Xt
        const Xt = XtreinoUns.transpose()
        // XtX
        const XtX = tf.matMul( Xt, XtreinoUns )
        // I
        const I = tf.tensor( [[1,0,0],[0,1,0],[0,0,1]] )
        // λI
        const lambdaI = tf.mul( lambda[0], I );
        // (Xt*X + λI)^-1
        const XtXlambdaInv = tf.tensor(
            math.inv(
                tf.add( XtX, lambdaI ).arraySync()
            )
        ) 
                
        // Garbage collection
        Xtreino.dispose();
        XtreinoUns.dispose();

        // XtY
        const XtY = tf.matMul(Xt, yTreino)

        // Garbage collection
        lambdaI.dispose();
        Xt.dispose();
        yTreino.dispose();

        const W = tf.matMul(XtXlambdaInv, XtY)
        W.print()
        // Garbage collection
        XtY.dispose();
        XtXlambdaInv.dispose();
        
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

        for (let i = 0; i < XtesteUns.shape[0]; i++) {
            let previsao = tf.matMul( tf.tensor([XtesteUns.arraySync()[i]]), W ).arraySync()[0]
            let real = yTeste.arraySync()[i]

            // Aplica o degrau unitario nos valores obtidos 
            // com o modelo e gera a matriz de confusao
            if (unitStep(previsao) == real){
                (real == 0 ? VN++ : VP++)
            } else {
                (real == 0 ? FP++ : FN++)
            }    
            
            // Garbage collection
            previsao = null
            real = null;
        }

        // Garbage collection
        Xteste.dispose();
        XtesteUns.dispose();
        yTeste.dispose();
        
        //const matrizConfusao = tf.tensor( [[VP, FP],[FN, VN]] )
        //matrizConfusao.print()

        acuracia.push( (VP+VN) / (VP+VN+FP+FN) )
        sensibilidade.push( (VP) / (VP+FN) )
        especificidade.push( (VN) / (VN+FP) )
    }

    console.log(acuracia)
    console.log(sensibilidade)
    console.log(especificidade)

    // Garbage Collection
    tensorFX1 = null;
    tensorFX2 = null;
    tensorFY = null;
    
    //One step is missing, before implementing the cost function. 
    //The input matrix X needs to add an intercept term. 
    //Only that way the matrix operations work for the dimensions of theta and matrix X.
    //X = tf.concat([tf.ones([m, 1]), X], 1);

    // Plota o gráfico
    //scatterPlot(emgObject);

    // Gera arquivo CSV com os dados estatisticos:
    // Acuracia, sensibilidade e especificidade
    let Data = [acuracia, sensibilidade, especificidade]
    let length = acuracia.length
    geraCSVcomDownload(Data, length)

    // Garbage Collection
    acuracia = null;
    sensibilidade = null;
    especificidade = null;
    emgObject = null;

    //closeWindow();

}

function lambdaLinearRegression(data){

    let seed = 0;

    let arAuxGr0 = [];
    let arAuxGr1 = [];
    let arAuxSu0 = [];
    let arAuxSu1 = [];

    for (const rod in data) {
        if(rod == "Rodada6") break;
        if (Object.hasOwnProperty.call(data, rod)) {
            arAuxGr0 = arAuxGr0.concat(data[rod].Grumpy[0])
            arAuxGr1 = arAuxGr1.concat(data[rod].Grumpy[1])   
            arAuxSu0 = arAuxSu0.concat(data[rod].Surpreso[0])
            arAuxSu1 = arAuxSu1.concat(data[rod].Surpreso[1])        
        }
    }

    // normalização dos dados
    const arAuxX = normalize( arAuxGr0.concat(arAuxSu0) )
    arAuxGr0 = arAuxX.slice(0, arAuxGr0.length ) 
    arAuxSu0 = arAuxX.slice(arAuxSu0.length, arAuxX.length)

    const arAuxY = normalize( arAuxGr1.concat(arAuxSu1) )
    arAuxGr1 = arAuxY.slice(0, arAuxGr1.length ) 
    arAuxSu1 = arAuxY.slice(arAuxSu1.length, arAuxY.length)
        
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

    // Garbage collection
    Y_grumpy.dispose();

    const Y_surpreso = tf.zeros([1, emgObject.surpreso[0].length*2]);
    const surpresoFinal = tf.tensor2d([emgObject.surpreso[0], emgObject.surpreso[1], Y_surpreso.arraySync()]).transpose();
    //   X1    X2   Y
    // [592 , 529 , 0],
    // [573 , 510 , 0],
    // [554 , 496 , 0], ...
    
    // Garbage collection
    Y_surpreso.dispose();
    
    // Dados são concatenados verticalmente
    const axis = 0;
    const finalTensor = grumpyFinal.concat(surpresoFinal, axis)

    // Garbage collection
    surpresoFinal.dispose();
    grumpyFinal.dispose();

    let tensorFX1 = normalize( finalTensor.transpose().arraySync()[0] );
    let tensorFX2 = normalize( finalTensor.transpose().arraySync()[1] );
    let tensorFY = finalTensor.transpose().arraySync()[2] ;

    // Garbage collection
    finalTensor.dispose();

    const treino80 = tensorFX1.length*.8
    const teste20 = tensorFX1.length*.2
  
    let rodadas = 100
    let acuracia = []
    let sensibilidade = []
    let especificidade = []
    
    for (let i = 0; i < rodadas; i++) {
            
        // Dados sao embaralhados
        let aux0 = shuffle( tensorFX1, seed )
        let aux1 = shuffle( tensorFX2, seed )
        let auxY = shuffle( tensorFY, seed )
        seed++;  
        
        // Divide 80% para treino e 20% para teste
        const Xtreino = tf.tensor( [aux0.slice(0,treino80), aux1.slice(0,treino80)] ).transpose()
        //Xtreino.print(true)
        const yTreino = tf.tensor( [auxY.slice(0,treino80)] ).transpose()
        //yTreino.print(true)
        const Xteste = tf.tensor( [aux0.slice(-teste20), aux1.slice(-teste20)] ).transpose()
        //Xteste.print(true)
        const yTeste = tf.tensor( [auxY.slice(-teste20)] ).transpose()
        //yTeste.print(true)

        // Calculo de W = (Xt*X + λI)^-1 * XtY 
        // λ
        const lambda = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.9,1]
        // Xt
        const Xt = Xtreino.transpose()
        // XtX
        const XtX = tf.matMul( Xt, Xtreino )
        // I
        const I = tf.tensor( [[1,0],[0,1]] )
        // λI
        const lambdaI = tf.mul( lambda[9], I );
        // (Xt*X + λI)^-1
        const XtXlambdaInv = tf.tensor(
            math.inv(
                tf.add( XtX, lambdaI ).arraySync()
            )
        ) 
                
        // Garbage collection
        Xtreino.dispose();

        // XtY
        const XtY = tf.matMul(Xt, yTreino)

        // Garbage collection
        lambdaI.dispose();
        Xt.dispose();
        yTreino.dispose();

        const W = tf.matMul(XtXlambdaInv, XtY)
        //W.print(true)

        // Garbage collection
        XtY.dispose();
        XtXlambdaInv.dispose();
        
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
            
            // Garbage collection
            previsao = null
            real = null;
        }

        // Garbage collection
        Xteste.dispose();
        yTeste.dispose();
        
        //const matrizConfusao = tf.tensor( [[VP, FP],[FN, VN]] )
        //matrizConfusao.print()

        acuracia.push( (VP+VN) / (VP+VN+FP+FN) )
        sensibilidade.push( (VP) / (VP+FN) )
        especificidade.push( (VN) / (VN+FP) )
    }

    console.log(acuracia)
    console.log(sensibilidade)
    console.log(especificidade)

    // Garbage Collection
    tensorFX1 = null;
    tensorFX2 = null;
    tensorFY = null;
    
    //One step is missing, before implementing the cost function. 
    //The input matrix X needs to add an intercept term. 
    //Only that way the matrix operations work for the dimensions of theta and matrix X.
    //
    //3. Em sala foi realizada uma discussao sobre a adicao de um vetor 
    //coluna de 1s no inicio da matriz de dados X.
    //X = tf.concat([tf.ones([m, 1]), X], 1);

    // Plota o gráfico
    scatterPlot(emgObject);

    // Gera arquivo CSV com os dados estatisticos:
    // Acuracia, sensibilidade e especificidade
    let Data = [acuracia, sensibilidade, especificidade]
    let length = acuracia.length
    geraCSVcomDownload(Data, length)

    // Garbage Collection
    acuracia = null;
    sensibilidade = null;
    especificidade = null;
    emgObject = null;

}

function linearRegression(data){

    let seed = 0;

    let arAuxGr0 = [];
    let arAuxGr1 = [];
    let arAuxSu0 = [];
    let arAuxSu1 = [];

    for (const rod in data) {
        if(rod == "Rodada6") break;
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

    // Garbage collection
    Y_grumpy.dispose();

    const Y_surpreso = tf.zeros([1, emgObject.surpreso[0].length*2]);
    const surpresoFinal = tf.tensor2d([emgObject.surpreso[0], emgObject.surpreso[1], Y_surpreso.arraySync()]).transpose();
    //   X1    X2   Y
    // [592 , 529 , 0],
    // [573 , 510 , 0],
    // [554 , 496 , 0], ...
    
    // Garbage collection
    Y_surpreso.dispose();
    
    // Dados são concatenados verticalmente
    const axis = 0;
    const finalTensor = grumpyFinal.concat(surpresoFinal, axis)

    // Garbage collection
    surpresoFinal.dispose();
    grumpyFinal.dispose();

    let tensorFX1 = finalTensor.transpose().arraySync()[0];
    let tensorFX2 = finalTensor.transpose().arraySync()[1];
    let tensorFY = finalTensor.transpose().arraySync()[2];

    // Garbage collection
    finalTensor.dispose();

    const treino80 = tensorFX1.length*.8
    const teste20 = tensorFX1.length*.2
  
    let rodadas = 1
    let acuracia = []
    let sensibilidade = []
    let especificidade = []
    
    for (let i = 0; i < rodadas; i++) {
            
        // Dados sao embaralhados
        let aux0 = shuffle( tensorFX1, seed )
        let aux1 = shuffle( tensorFX2, seed )
        let auxY = shuffle( tensorFY, seed )
        seed++;  
        
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
        
        tf.matMul( Xt, Xtreino ).print(true)
        
        // Garbage collection
        Xtreino.dispose();

        // XtY
        const XtY = tf.matMul(Xt, yTreino)

        // Garbage collection
        Xt.dispose();
        yTreino.dispose();

        const W = tf.matMul(XtXinv, XtY)

        // Garbage collection
        XtY.dispose();
        XtXinv.dispose();
        
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
        
        //const matrizConfusao = tf.tensor( [[VP, FP],[FN, VN]] )
        //matrizConfusao.print()

        acuracia.push( (VP+VN) / (VP+VN+FP+FN) )
        sensibilidade.push( (VP) / (VP+FN) )
        especificidade.push( (VN) / (VN+FP) )
    }

    console.log(acuracia)
    console.log(sensibilidade)
    console.log(especificidade)

    // Garbage Collection
    tensorFX1 = null;
    tensorFX2 = null;
    tensorFY = null;
    
    //One step is missing, before implementing the cost function. 
    //The input matrix X needs to add an intercept term. 
    //Only that way the matrix operations work for the dimensions of theta and matrix X.
    //
    //3. Em sala foi realizada uma discussao sobre a adicao de um vetor 
    //coluna de 1s no inicio da matriz de dados X.
    //X = tf.concat([tf.ones([m, 1]), X], 1);

    // Plota o gráfico
    scatterPlot(emgObject);

    // Gera arquivo CSV com os dados estatisticos:
    // Acuracia, sensibilidade e especificidade
    let Data = [acuracia, sensibilidade, especificidade]
    let length = acuracia.length
    geraCSVcomDownload(Data, length)

}

getJsonData = async (filePath) => {
    const response = await fetch(filePath)
    const data = await response.json();
    return data;
}

// Regressão linear nao-regularizada
// getJsonData('./emg.json').then((res) => linearRegression(res));

// Regressão linear regularizada
//getJsonData('./emg.json').then((res) => lambdaLinearRegression(res));

// Regressão linear regularizada com uns
//getJsonData('./emg.json').then((res) => lambdaLinearRegressionOnes(res));

// Perceptron Simples
getJsonData('./emg.json').then((res) => simplePerceptron(res));
