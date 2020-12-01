function dataPull(data) {
    var output = data
    return output
};

// dropdown functions
function systemsDropdown(output) {
    var systems = output.systems;
    var systemsMenu = d3.select("#consolesDropdown");
    systemsMenu
        .selectAll("option")
        .enter()
        .data(systems)
        .enter()
        .append("option")
        .attr("value", function(d){
            return d;
        })
        .text(function(d){
            return d;
        });
    return systemsMenu
  };

function genresDropdown(output) {
    var genres = output.genres;
    var genresMenu = d3.select("#genresDropdown");
    genresMenu
        .selectAll("option")
        .enter()
        .data(genres)
        .enter()
        .append("option")
        .attr("value", function(d){
            return d;
        })
        .text(function(d){
            return d;
        });
    return genresMenu
  };

function themesDropdown(output) {
    var themes = output.themes;
    var themesMenu = d3.select("#themesDropdown");
    themesMenu
        .selectAll("option")
        .enter()
        .data(themes)
        .enter()
        .append("option")
        .attr("value", function(d){
            return d;
        })
        .text(function(d){
            return d;
        });
    return themesMenu
  };

function perspectivesDropdown(output) {
    var perspectives = output.playerPerspectives;
    var perspectivesMenu = d3.select("#perspectiveDropdown");
    perspectivesMenu
        .selectAll("option")
        .enter()
        .data(perspectives)
        .enter()
        .append("option")
        .attr("value", function(d){
            return d;
        })
        .text(function(d){
            return d;
        });
    return perspectivesMenu
  };

function playModesDropdown(output) {
    var playModes = output.playModes;
    var playModesMenu = d3.select("#playModesDropdown");
    playModesMenu
        .selectAll("option")
        .enter()
        .data(playModes)
        .enter()
        .append("option")
        .attr("value", function(d){
            return d;
        })
        .text(function(d){
            return d;
        });
    return playModesMenu
  };

function seriesDropdown(output) {
    var series = output.series;
    var seriesMenu = d3.select("#seriesDropdown");
    seriesMenu
        .selectAll("option")
        .enter()
        .data(series)
        .enter()
        .append("option")
        .attr("value", function(d){
            return d;
        })
        .text(function(d){
            return d;
        });
    return seriesMenu
  };