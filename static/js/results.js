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

// bar chart functions
// Define SVG area dimensions
var svgWidth = 900;
var svgHeight = 500;

// Define the chart's margins as an object
var margin = {
    top: 40,
    right: 40,
    bottom: 200,
    left: 150
  };

// // Define dimensions of the chart area
// var chartWidth = svgWidth - margin.left - margin.right;
// var chartHeight = svgHeight - margin.top - margin.bottom;

// var svg = d3
//   .select("#resultPlot")
//   .append("svg")
//   .attr("height", svgHeight)
//   .attr("width", svgWidth);

//   function createBars(output) {
//       // star data
//       var starData = output.allRatings
//       // result rating
//       var prediction = output.prediction

//       // Append a group area, then set its margins
//       var chart = svg.append("g")
//       .attr("transform", `translate(${margin.left}, ${margin.top})`);
      
//       chartGroup.selectAll(".bar")
//       .data(starData)
//       .enter()
//       .append("rect")
//       .classed("bar", true)
//       .attr("width", d => barWidth)
//       .attr("height", d => d.hours * scaleY)
//       .attr("x", (d, i) => i * (barWidth + barSpacing))
//       .attr("y", d => chartHeight - d.hours * scaleY);  
  
//       var yLinearScale = d3.scaleLinear()
//         .range([chartHeight, 0])
//         .domain([0, d3.max(graphData, function(d) {
//           return d[method];
//         })]);
  