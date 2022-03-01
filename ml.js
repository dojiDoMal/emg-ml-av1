// https://www.freecodecamp.org/news/the-least-squares-regression-method-explained/
// https://pt.wikipedia.org/wiki/M%C3%A9todo_dos_m%C3%ADnimos_quadrados

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

function handleData(data, rodada){

    const seed = 15;
    const dadosRodada = data[rodada];
    let emgObject = new EMG(dadosRodada);

    // Embaralha os valores
    emgObject.grumpy[0] = shuffle(emgObject.grumpy[0], seed);
    emgObject.grumpy[1] = shuffle(emgObject.grumpy[1], seed);

    emgObject.surpreso[0] = shuffle(emgObject.surpreso[0], seed);
    emgObject.surpreso[1] = shuffle(emgObject.surpreso[1], seed);

    // Cria os modelos de regressão linear
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

    // Cria os tensores
    const grumpySensor1 = tf.tensor2d(emgObject.grumpy[0], [emgObject.grumpy[0].length, 1], 'int32');
    const grumpySensor2 = tf.tensor2d(emgObject.grumpy[1], [emgObject.grumpy[1].length, 1], 'int32');

    const surpresoSensor1 = tf.tensor2d(emgObject.surpreso[0], [emgObject.surpreso[0].length, 1], 'int32');
    const surpresoSensor2 = tf.tensor2d(emgObject.surpreso[1], [emgObject.surpreso[1].length, 1], 'int32');

    grumpySensor1.print();
    grumpySensor2.print();

    surpresoSensor1.print();
    surpresoSensor2.print();

    // Plota o gráfico
    scatterPlot(emgObject, {traceLine: true, lines: [grumpyLine, surpresoLine]});
}

getJsonData = async (filePath) => {
    const response = await fetch(filePath)
    const data = await response.json();
    return data;
}

getJsonData('./emg.json').then((res) => handleData(res, EnumRodadas.R1));

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
