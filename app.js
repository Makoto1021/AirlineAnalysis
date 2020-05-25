/*
require('dotenv').config();
console.log(process.env);

const fetch = require('node-fetch');
global.Headers = fetch.Headers;
*/
let defaultBackGroundColors = ['rgb(252, 66, 123, 0.5)', 'rgb(22, 160, 133, 0.5)','rgb(52, 152, 219, 0.5)','rgb(155, 89, 182, 0.5)',
                      'rgb(126, 214, 223, 0.5)','rgb(59, 62, 172, 0.5)','rgb(0, 153, 198, 0.5)','rgb(221, 68, 119, 0.5)',
                      'rgb(102, 170, 0, 0.5)','rgb(184, 46, 46, 0.5)','rgb(49, 99, 149, 0.5)','rgb(153, 68, 153, 0.5)',
                      'rgb(34, 170, 153, 0.5)','rgb(170, 170, 17, 0.5)','rgb(102, 51, 204, 0.5)','rgb(230, 115, 0, 0.5)',
                      'rgb(139, 7, 7, 0.5)','rgb(50, 146, 98, 0.5)','rgb(85, 116, 166, 0.5)','rgb(59, 62, 172, 0.5)'];

let defaultBorderColors = ['rgb(252, 66, 123)', 'rgb(22, 160, 133)','rgb(52, 152, 219)','rgb(155, 89, 182)',
                      'rgb(126, 214, 223)','rgb(59, 62, 172)','rgb(0, 153, 198)','rgb(221, 68, 119)',
                      'rgb(102, 170, 0)','rgb(184, 46, 46)','rgb(49, 99, 149)','rgb(153, 68, 153)',
                      'rgb(34, 170, 153)','rgb(170, 170, 17)','rgb(102, 51, 204)','rgb(230, 115, 0)',
                      'rgb(139, 7, 7)','rgb(50, 146, 98)','rgb(85, 116, 166)','rgb(59, 62, 172)'];

setFilterOptions();

console.log($(document).ready(function() {
 $("#companies").select2();
}))

function onBarClicked(company, category) {
    /* function to get noun tables after a user clicks the bar chart */
    
    let headers = new Headers();
    let init = {
        method : 'GET',
        headers : headers
    };
    let params = {
        'company':company,
        'category':category
    };
    console.log("stringify json", JSON.stringify(params))
    
    // let url = getWebAppBackendUrl('/get_table')+'/'+JSON.stringify(params);
    let url = 'http://127.0.0.1:7990'+'/get_table' + '/'+JSON.stringify(params); 
    let promise = fetch(url, init) 
                   .then(function(response) {
                       response.json()
                       .then(function(data){
                           let title_pos = 'Top 5 positive sentiment contributors for ' + company + " in " + category;
                           let title_neg = 'Top 5 negative sentiment contributors for ' + company + " in " + category;
                           document.getElementById('title_pos').innerHTML = title_pos;
                           document.getElementById('title_neg').innerHTML = title_neg;
                           addRows("insert_table_pos", data["table_pos"]);
                           addRows("insert_table_neg", data["table_neg"]);
                           addRowHandlers("insert_table_pos", data["table_pos"], company, category);
                           addRowHandlers("insert_table_neg", data["table_neg"], company, category);
                       })
                   });
};


function addRowHandlers(id, data, company, category) {
    /* function to add an action when a user clicks nouns in the tables */
    /* this function adds table of raw tweets and a noun chart */
    
    var table = document.getElementById(id);
    var rows = table.getElementsByTagName("tr");
    for (i = 0; i < rows.length; i++) {
        row = table.rows[i];
        row.onclick = function(){
            let cell = this.getElementsByTagName("td")[0];
            let topic = cell.innerHTML;
            console.log("clicked topic", topic);
            
            let dataJson=JSON.parse(data)
            console.log("Object.keys(dataJson)", Object.keys(dataJson))
            const found_key = Object.keys(dataJson).find(key => dataJson[key]['topics'] === topic);
            console.log("found_key", found_key)
            let review_id = dataJson[found_key]['review_id']
            // console.log("review_id", review_id)
            
            let headers = new Headers();
            let init = {
                method : 'GET',
                headers : headers
            };
            let params = {
                'company':company,
                'category':category,
                'review_id':review_id,
                'topic':topic
            };
            
            // This is to add tweets tables
            // let url1 = getWebAppBackendUrl('/get_tweets_table')+'/'+JSON.stringify(params);
            let url = 'http://127.0.0.1:7990' + '/get_tweets_table' +'/'+JSON.stringify(params);
            let promise1 = fetch(url1, init).then(function(response) {
                response.json()
                    .then(function(data){
                    console.log("this is how tweets look like", data['0'])
                    // document.getElementById('tweet_table').innerHTML = data;
                    let shortMessage = "To refresh the sample tweets, click the noun again";
                    document.getElementById("card-text-message").innerHTML = shortMessage;
                    let titleTweetsTable = 'Randomly selected tweets about "' + topic + '" for ' + company;
                    document.getElementById('title_tweets').innerHTML = titleTweetsTable;
                    addTweetRows('insert_tweet_table', data);
                });
            });
            
            // This is to add a sentiment comparison chart
            let url2 = getWebAppBackendUrl('/compare_companies_from_nouns')+'/'+JSON.stringify(params);
            let promise2 = fetch(url2, init).then(function(response) {
                response.json()
                    .then(function(data){
                    console.log("this is how noun chart data looks like", data.barNounChart)
                    let titleNounChart = 'Sentiment comparison for "' + topic + '" among companies ';
                    document.getElementById('title_noun_chart').innerHTML = titleNounChart;
                    addNounChart('noun-chart', data.barNounChart);
                    
                });
            });
        };
    };
};

/* returns the company value chosen from the dropdowns */
document.getElementById('button').onclick = function(){
    // let company = getSelectedOption('companies');
    let companies = $("#companies").val();
    console.log("company is", companies)
    let headers = new Headers()
    let init = {
        method : 'GET',
        headers : headers
    }
    let params = {
        'companies':companies
    }

    // let url = getWebAppBackendUrl('/get_stats')+'/'+JSON.stringify(params)
    let url = 'http://127.0.0.1:7990' + '/get_stats' + '/'+JSON.stringify(params)
    let promise = fetch(url, init) 
                   .then(function(response){
                       response.json()
                               .then(function(data){
                                // addBarChart(data.chart.data, data.chart.labels);
                           console.log("data.barChartGroup.company", data.barChartGroup.company)
                           console.log("data.barChartGroup.labels", data.barChartGroup.labels)
                                addGroupBarChart(data.barChartGroup.company, 
                                                 data.barChartGroup.labels);
                                })
                   });
}



function getSelectedOption(id){
/* function to get the selected value from the dropdowns */
    return $("#" + id).val();
}


function setFilterOptions(){
    /* function to get distinct values of companies and insert them in the dropdowns */
    let headers = new Headers()
    let init = {
        method : 'GET',
        headers : headers
    }

    // let url = getWebAppBackendUrl('/get_filter_values')
    let url = 'http://127.0.0.1:7990' + '/get_filter_values'
    let promise = fetch(url, init)
                   .then(function(response){
                       response.json()
                               .then(function(data){
                                let companiesList = data.companies;
                                addOptions('companies', companiesList);                                
                                })
                   });
                       
}

function addRows(id, itemList){
    /* function to add rows to a table */
    let itemJson=JSON.parse(itemList)
    console.log("itemJson", typeof(itemJson))
    document.getElementById(id).innerHTML = '';
    i = 0
    Object.keys(itemJson).forEach(function(key) {
        // this iterates over rows
        // console.table('Key : ' + key + ', Value : ' + itemJson[key]);
        let topics = itemJson[key]["topics"]
        let averageScores = itemJson[key]["average scores"]
        let importanceScores = itemJson[key]["importance scores"]
        // console.log("topics", topics)
        // console.log("averageScores", averageScores)
        // console.log("importanceScores", importanceScores)
        let tbl = document.getElementById(id)        
        let row = tbl.insertRow(i)
        row.id = "rowID_" + i;
        let cell1 = row.insertCell(0);
        let cell2 = row.insertCell(1);
        let cell3 = row.insertCell(2);
        cell1.innerHTML = topics;
        cell2.innerHTML = averageScores;
        cell3.innerHTML = importanceScores;
        i += 1
})
    var orderArrayHeader = ["Topics", "Average sentiment","Contribution score"];
    var thead = document.createElement('thead');
    let finishedTable = document.getElementById(id)   
    finishedTable.appendChild(thead);
    for(var j=0;j<orderArrayHeader.length;j++){
        thead.appendChild(document.createElement("th")).
        appendChild(document.createTextNode(orderArrayHeader[j]));
    }
    
    

    
    let tblToClick = document.getElementById(id);
    for (let i = 0; i < tblToClick.rows.length; i++) {
        for (let j = 0; j < tblToClick.rows[i].cells.length; j++) {
            tblToClick.rows[i].cells[j].onclick = (function (i, j) {
                return function () {
                    // alert('R' + (i + 1) + 'C' + (j + 1));
                };
            }(i, j));
        }
    }
}

function addTweetRows(id, itemList){
    /* function to add rows to a table */
    console.log("itemList", typeof(itemList));
    if (typeof(itemList) != 'object') {
        let itemJson=JSON.parse(itemList);
    } else {
        itemJson = itemList
    }
    
    document.getElementById(id).innerHTML = '';
    let orderArrayHeader = []
    i = 0
    Object.keys(itemJson).forEach(function(key) {
        // this iterates over rows
        // console.table('Key : ' + key + ', Value : ' + itemJson[key]);
        let tbl = document.getElementById(id);
        let row = tbl.insertRow(i);
        row.id = "rowID_" + i;
        
        j=0
        arrHyperLink = []
        let user_id = itemJson[key]["user_id"];
        let tweet_id = itemJson[key]["tweet_id"];
        let hyperLink = "https://twitter.com/" + user_id + "/status/" + tweet_id;

        Object.keys(itemJson[key]).forEach(function(sub_key){
            console.log("sub_key is", sub_key)
            cellValue = itemJson[key][sub_key];
            let cell = row.insertCell(j);
            if (sub_key==="tweet_id") {
                cell.innerHTML = '<a href="'+ hyperLink +'" target="_blank">'+ cellValue +'</a>';
            } else {
                cell.innerHTML = cellValue;
            }
            if (orderArrayHeader.length < Object.keys(itemJson[key]).length){
                orderArrayHeader.push(sub_key)
            }
            j+=1
        })
        i += 1
})
    
    let thead = document.createElement('thead');
    let finishedTable = document.getElementById(id);   
    finishedTable.appendChild(thead);
    for(let j=0;j<orderArrayHeader.length;j++){
        thead.appendChild(document.createElement("th")).
        appendChild(document.createTextNode(orderArrayHeader[j]));
    }
}

function addOptions(id, itemList){
    /* function to add company choice to the dropdowns */
    itemList.forEach(function(value, index){
        let option = document.createElement('option');
        option.text = value;
        document.getElementById(id).appendChild(option)
    })
}

function addGroupBarChart(data, labels) {
    /* function to add a bar chart for a selected companies */
    document.getElementById('bar-card').innerHTML = '<canvas id="bar-chart-group" chart-click="onClick"></canvas>';
    i = 1
    arr = []
    console.log("labels", labels)
    data.forEach(function(d){
        company = Object.keys(d);
        values = d[company]['data'];
        dataset_company = {
            label:company,
            backgroundColor: defaultBackGroundColors[i],
            borderColor: defaultBorderColors[i],
            borderWidth: 1,
            data:values};
        console.log("dataset_company", dataset_company);
        arr.push(dataset_company);
        i+=1;
    })
    
    let data_grouped = {
        labels: labels,
        datasets: arr
              };
    
    // this.groupBarChart = 
    var chart = new Chart(document.getElementById("bar-chart-group"), {
        type: "bar",
        data: data_grouped,
        options: {
          title: {
            display: false,
            text: "Sentiment by category"
          },
            onClick: event => {
                const datasetIndex = chart.getElementAtEvent(event)[0]._datasetIndex;
                const model = chart.getElementsAtEvent(event)[datasetIndex]._model;
                onBarClicked(model.datasetLabel, model.label);
            }
        }
    });
}

function addNounChart(id, data) {
    /* function to add a bar chart for a selected noun */
    
    document.getElementById('noun-chart').innerHTML = '<canvas id="noun-chart-group" chart-click="onClick"></canvas>';
    Chart.defaults.global.legend.display = false;
    
    var labels = data.map(function(e) {
        return e.product_id;
    });
    
    var values = data.map(function(e) {
        return e.mean_polarity_textblob;
    });;
    
    var myBarChart = new Chart(document.getElementById("noun-chart-group"), {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: defaultBackGroundColors,
                borderColor: defaultBorderColors,
                borderWidth: 1
            }]
        },
        options: {
            title: {
                display: false,
                text: "Sentiment comparison"
            }
        }
    });
}