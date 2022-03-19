// Enum para organizar e separar as rodadas contidas nos dados
function EnumRodadas(){}
EnumRodadas.R1 = 'Rodada1';
EnumRodadas.R2 = 'Rodada2';
EnumRodadas.R3 = 'Rodada3';
EnumRodadas.R4 = 'Rodada4';
EnumRodadas.R5 = 'Rodada5';
EnumRodadas.R6 = 'Rodada6';
EnumRodadas.R7 = 'Rodada7';
EnumRodadas.R8 = 'Rodada8';
EnumRodadas.R9 = 'Rodada9';
EnumRodadas.R10 = 'Rodada10';

/**
 * Y = bX + a 
 * 
 * Onde:
 * 
 * b = (Σ(x - x') * (y - y')) / Σ(x - x')^2
 * 
 * x', y' = valor médio
 * @param {number[]} x 
 * @param {number[]} y 
 * @returns {object} Objeto contendo a, b, x', y', (x-x'), (y-y')
 */
 function calculateLinearRegression(x, y){
    
    // x'
    let xAvg = calculateAverage(x);
    // y'
    let yAvg = calculateAverage(y);

    // (x - x')
    let xDiff = calculateAverageDiff(x, xAvg);
    // (y - y')
    let yDiff = calculateAverageDiff(y, yAvg);

    // b
    let sumNum = 0;
    let sumDen = 0;
    let b = 0;
    for (let i = 0; i < x.length; i++) {
        sumNum += xDiff[i]*yDiff[i];
        sumDen += xDiff[i]*xDiff[i];        
    }  
    b = sumNum / sumDen;

    // a
    let a = 0
    a = yAvg - b*xAvg;

    return {
        a: a,
        b: b,
        xAverage: xAvg,
        yAverage: yAvg,
        xDifference: xDiff,
        yDifference: yDiff
    }
}

function costFunction(theta, X, y){
    const m = y.length;

    //let h = sigmoid(X * theta) //Acho que aqui e um produto interno

    // ' = transposto
    // (1 / m) * (-y' * log(h) - (1 - y)' * log(1 - h))
    //const cost = (1/m)*(-y)
    //return cost
}

function sigmoid(z){
    return 1/(1+Math.exp(-z));
}

function unitStep(x){
    return x >= 0.5 ? 1 : 0
    //return 1 * (x > 0);
}

function calculateAverage(array){
    let sum = 0;
    let average = 0;
    for (let i = 0; i < array.length; i++) {
        sum += array[i]; 
    }
    average = sum / array.length;
    return average;
}

function calculateAverageDiff(array, average){
    let diff = [];
    for (let i = 0; i < array.length; i++) {
        diff.push(array[i] - average);        
    }
    return diff;
}

/**
 * Permite embaralhar um array de forma determinável por meio de uma seed.
 * @param {array} array O array que deverá ser embaralhado 
 * @param {number} seed O valor da seed
 * @returns {array} O array embaralhado
 */
function shuffle(array, seed) {                
    var m = array.length, t, i;
    while (m) {
        i = Math.floor(random(seed) * m--);   
        t = array[m];
        array[m] = array[i];
        array[i] = t;
        ++seed                                     
    }
    return array;
}
 
/**
 * Gera um numero aleatório a partir de uma seed
 * @param {number} seed O valor da seed
 * @returns {number} Um número aleatório gerado com base na seed 
 */ 
function random(seed) {
    var x = Math.sin(seed++) * 10000; 
    return x - Math.floor(x);
}

// Função responsável por plotar o grafico com os dados na tela
function scatterPlot(objData){
    const traceGrumpy = {
        x: objData.grumpy[0],
        y: objData.grumpy[1],
        mode: 'markers',
        type: 'scatter',
        name: 'Grumpy',
        marker: {size: 5}       
    }

    const traceSurpreso = {
        x: objData.surpreso[0],
        y: objData.surpreso[1],
        mode: 'markers',
        type: 'scatter',
        name: 'Surpreso',
        marker: {size: 5}       
    }

    let x1 = []
    let x2 = []

    const w1 = objData.params.w1
    const w2 = objData.params.w2
    const theta = objData.params.theta
    console.log(objData)

    for (let i = 0; i < 3000; i++) {
        x1.push(i)
        x2.push( (-(w1/w2)*i) + (theta/w2) )        
    } 
    
    const linhaClasses = {
        x: x1,
        y: x2,
        type: 'scatter',
        name: 'Separador',
        marker: {size: 3}    
    }

    const layout = {
        xaxis: {
            autorange: false,
            range: [0,3000]
        },
        yaxis: {
            autorange: false,
            range: [0,2500]
        }    
    }

    const traceData = [traceGrumpy, traceSurpreso, linhaClasses];   

    Plotly.newPlot('plot', traceData, layout, {responsive: true})
    
}