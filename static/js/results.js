function dataPull(data) {
    var output = data
    return output
};

function convertNumToStar(data) {
    if (data > 85) {
        return 5;
    } else if (data > 75) {
        return 4;
    } else if (data > 65) {
        return 3;
    } else if (data > 55) {
        return 2;
    } else {
        return 1;
    }
};
function colorString(prediction) {
    console.log(prediction)
    if (prediction === 1) {
        return ['rgb(116, 182, 82)', 'rgb(101, 52, 150)', 'rgb(101, 52, 150)', 'rgb(101, 52, 150)', 'rgb(101, 52, 150)'];
    } else if (prediction === 2) {
        return ['rgb(101, 52, 150)', 'rgb(116, 182, 82)', 'rgb(101, 52, 150)', 'rgb(101, 52, 150)', 'rgb(101, 52, 150)'];
    } else if (prediction === 3) {
        return ['rgb(101, 52, 150)', 'rgb(101, 52, 150)', 'rgb(116, 182, 82)', 'rgb(101, 52, 150)', 'rgb(101, 52, 150)'];
    } else if (prediction === 4) {
        return ['rgb(101, 52, 150)', 'rgb(101, 52, 150)', 'rgb(101, 52, 150)', 'rgb(116, 182, 82)', 'rgb(101, 52, 150)'];
    } else {
        return ['rgb(101, 52, 150)', 'rgb(101, 52, 150)', 'rgb(101, 52, 150)', 'rgb(101, 52, 150)', 'rgb(116, 182, 82)'];
    }
};