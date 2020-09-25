const axios = require('axios')
var http = require('http');


async function get_webService(msg) {
    return axios
        .post('http://127.0.0.1:5000/api/resultat', {
            "trading": msg ,   
        })
        .then(res => {
             console.log(res.data);
            return res.data
        })
 
}

async function Malek(){

    return 'hello'
}