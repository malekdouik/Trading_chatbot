const axios = require('axios')
var http = require('http');


const hostname = '127.0.0.1';
const port = 3000;

//Partie Sami 1
async function get_webService_Price(msg,a) {
    console.log("aaaaaaaaaaaaaaaaaaaaaaa"+a)
    return axios
        .post(`http://0.0.0.0:5000/model${a}?crypto=${a}`, {
            "trading": msg ,   
        })
        .then(res => {
            console.log("WebService_File----------------")
            console.log(res.data);
            return res.data
        })        
        .catch(error => {
            console.error(error)
        })   
}

// Partie 2 Sami: crypto
async function get_webService_Cryto_real(msg,a) {
    console.log("aaaaaaaaaaaaaaaaaaaaaaa"+a)
    return axios
        .post(`http://0.0.0.0:5000/realCrypto?crypto=${a}`, {
            "trading": msg ,   
        })
        .then(res => {
            console.log("WebService_File----------------")
            console.log(res.data);
            return res.data
        })        
        .catch(error => {
            console.error(error)
        })   
}



// Partie 2 Sami: forex
async function get_webService_forex_real(msg,a) {
    console.log("aaaaaaaaaaaaaaaaaaaaaaa"+a)
    
    return axios
        .post(`http://0.0.0.0:5000/realForex?symbole1=${a}`, {
            "trading": msg ,   
        })
        .then(res => {
            console.log("WebService_File----------------")
            console.log(res.data);
            return res.data
        })        
        .catch(error => {
            console.error(error)
        })   
}



/*
async function get_webService_Actions_real(msg,a) {
    console.log("aaaaaaaaaaaaaaaaaaaaaaa"+a)
    return axios
        .post(`http://0.0.0.0:5000/realAction?crypto=${a}`, {
            "trading": msg ,   
        })
        .then(res => {
            console.log("WebService_File----------------")
            console.log(res.data);
            return res.data
        })        
        .catch(error => {
            console.error(error)
        })   
}
*/


// function to send data to Flask
/*
async function get_webServ_parameters(parameters) {
    
    var express = require('express');
    var http = require('http');
    
    var app = express();
    
    app.get('/', (req, res) => res.send('Hello World!'));
    
    app.listen(5000, () => console.log('Running on http://localhost:5000'));
    
    postData = JSON.stringify({
        'code': parameters
    });

    var options = {
        hostname: 'localhost',
        port: 5000,
        path: '/test',
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
            'Content-Length': Buffer.byteLength(postData)
        }
    };

    var req = http.request(options, (res) => {
        var data = '';
        res.on('data', (chunk) => {
            data += chunk.toString(); // buffer to string
        });
        res.on('end', () => {
            data = JSON.parse(data);
            console.log(data.message);
            console.log('No more data in response.');
        });
    });
    
    req.on('error', (e) => {
        console.error(`problem with request: ${e.message}`);
    });
    
    req.write(postData);
    req.end();

}
*/
//Malek
var externalip = require('external-ip');
async function get_parameters_powerBI(parameters) {
    
    // Send data to flask via AJAX
    console.log("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA : "+parameters)
    console.log(JSON.stringify(parameters))
    a = parameters.match(/\:"(.*?)\"}/i)[1]
    console.log("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ"+ a)

    var jsdom = require('jsdom');

    const { JSDOM } = jsdom;
    const { window } = new JSDOM();
    const { document } = (new JSDOM('')).window;
    global.document = document;
    
    var $ = jQuery = require('jquery')(window);
    
        
            $.ajax({
                url: "http://127.0.0.1:5000/powerbi",
                type: "POST",
                data: a,
                dataType: "json",
                success: function() {
                    console.log("success!")
                },
                error: function() {
                    console.log("error", arguments[2])
                }
            });
        }


///Malek statiques


async function get_webService_opt_portfolio(msg) {
    return axios
        .post('http://127.0.0.1:5000/test', {
            "trading": msg ,   
        })
        .then(res => {
            console.log(res.data);
            return res.data
        })        
        .catch(error => {
            console.error(error)
        }) 
}



module.exports={
    
    get_webService_Price : get_webService_Price,
    //get_webServ_parameters : get_webServ_parameters,
    get_webService_Cryto_real : get_webService_Cryto_real,
    get_webService_forex_real : get_webService_forex_real,
    get_parameters_powerBI : get_parameters_powerBI,
    get_webService_opt_portfolio : get_webService_opt_portfolio
 };
 