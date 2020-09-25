const dialogflow = require('dialogflow');
const uuid = require('uuid');
/* to link with the front end */
const express = require('express');
const bodyParser = require('body-parser');
const port = 5000;
const app = express();
/**/

const sessionId = uuid.v4();
//const mysql = require('mysql');
//const sessionId = uuid.v4();
var webservice = require('./webservice');

const {struct} = require('pb-util');



app.use(bodyParser.urlencoded({
  extended: false
}))


app.use(function (req, res, next) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, PATCH, DELETE');
  res.setHeader('Access-Control-Allow-Headers', 'X-Requested-With,content-type');
  res.setHeader('Access-Control-Allow-Credentials', true);
  // Pass to next layer of middleware
  next();
});

app.post('/send-msg',(req,res)=>{
runSample(req.body.MSG).then(data=>{
    res.send({Reply:data})
})
})

/**
 * Send a query to the dialogflow agent, and return the query result.
 * @param {string} projectId The project to be used
 */
async function runSample(msg,projectId = 'rn-bot-wcpx') {

  // Create a new session
  const sessionClient = new dialogflow.SessionsClient(
    {
      keyFilename:'./rn-bot-wcpx-80457c1f89a5.json'
    }
  );

  const sessionPath = sessionClient.sessionPath(projectId, sessionId);
  // The text query request.
  const request = {
    session: sessionPath,
    queryInput: {
      text: {
        // The query to send to the dialogflow agent
        text: msg,
        // The language used by the client (en-US)
        languageCode: 'en-US',
      },
    },
  };

  // Send request and log result
  const responses = await sessionClient.detectIntent(request);
  console.log('Detected intent');
  const result = responses[0].queryResult;
  console.log(`  Query: ${result.queryText}`);
  console.log(`  Response: ${result.fulfillmentText}`);
  
  if (result.intent) {
    console.log(`  Intent: ${result.intent.displayName}`);
  } else {
    console.log(`  No intent matched.`);
  }

  if (result.intent.displayName === 'price_tomorrow') {
    
    const parameters = JSON.stringify(struct.decode(result.parameters));
    console.log('CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC :' + parameters)
   
    a = parameters.match(/\:"(.*?)\"}/i)[1]
    console.log('sssssssssssssssssssssssssssssssss :' + a)
    console.log(msg)
    var res = await webservice.get_webService_Price(msg,a); 

    return JSON.stringify("Price Tomorrow is  :"+ res)
  }

  if (result.intent.displayName === 'price_crypto_today') {
    
    const parameters = JSON.stringify(struct.decode(result.parameters));
    console.log('CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC :' + parameters)
   
    a = parameters.match(/\:"(.*?)\"}/i)[1]
    console.log('sssssssssssssssssssssssssssssssss :' + a)
    console.log(msg)
    var res = await webservice.get_webService_Cryto_real(msg,a); 
    
    console.log('vvvvvvvvvvvvvvv'+ res['time'])
    t = JSON.stringify(res['time'])
    p = JSON.stringify(res['price today'])
    return ["time :" , t , "price cryptocurrency :", p]
  }

  if (result.intent.displayName === 'price_forex_today') {
    
    const parameters = JSON.stringify(struct.decode(result.parameters));
    console.log('CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC :' + parameters)
   
    a = parameters.match(/\:"(.*?)\"}/i)[1]
    console.log('sssssssssssssssssssssssssssssssss :' + a)
    console.log(msg)
    var res = await webservice.get_webService_forex_real(msg,a); 
    
    console.log('vvvvvvvvvvvvvvv'+ res['time'])
    t = JSON.stringify(res['time'])
    p = JSON.stringify(res['price today'])
    from = JSON.stringify(res['from_currency'])
    to = JSON.stringify(res['to_currency'])

    return [ "from : ", from , "to" , to , "time :" , t , "price exchange forex:", p ]
  }

// Feriel 
if (result.intent.displayName === 'dashbord') {
  var str = "Your dashbord";
  var lien = str.link('https://app.powerbi.com/groups/me/reports/780af077-fe5f-41c4-9e70-223ecab5d280/ReportSectionb58e90e19503ae85c757?redirectedFromSignup=1&noSignUpCheck=1&response=AlreadyAssignedLicense');
  return lien
}



//Maleeek
if (result.intent.displayName === 'dashboard_powerBI') {
  const parameters = JSON.stringify(struct.decode(result.parameters));
  var res = await webservice.get_parameters_powerBI(parameters); 

  a = parameters.match(/\:"(.*?)\"}/i)[1]
  
  var str = 'your Dashbord of:'+ a;
  var lien = str.link('https://app.powerbi.com/groups/me/reports/7c5de83c-6190-4e27-af21-ef193a5f5ae5/ReportSection?redirectedFromSignup=1&noSignUpCheck=1&response=AlreadyAssignedLicense');
  return lien
}


//Malek statique:
// Malek
if (result.intent.displayName === 'optimization_portfolio') {
    
  const parameters = JSON.stringify(struct.decode(result.parameters));
  console.log('the parameters detected : ' + parameters)
 
  //var res = await webservice.get_webServ_parameters(parameters); 
  var res2= await webservice.get_webService_opt_portfolio(msg)

  return JSON.stringify("The best portfolio is :"+ res2)
}
/*

if (result.intent.displayName === 'clustering') {

  var res2= await webservice.get_clustering(msg)
  return JSON.stringify("The proposed classification : "+ res2)

}
/*

//Exemple 
/*if (result.intent.displayName === 'other') {
  //console.log(msg)
  //var res = await webservice.get_webServ_parameters(msg);
  

  //var obj = JSON.parse(res);
  //console.log(obj['test']);
  //return "A "+ "'"+obj['test']+"' ."+"\ A hello";
  return 'Helooooo'
}
*/

return result.fulfillmentText;
}

//runSample()

app.listen(port, () => {

  console.log("running on port" + port)
})
