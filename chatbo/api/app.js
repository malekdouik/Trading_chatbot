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

  if (result.intent.displayName === 'trading') {
    console.log(msg)
    var res = await webservice.get_webService(msg);
    
    var text_json= JSON.stringify(res)

    var parsed = JSON.parse(text_json); 
    
    var user = parsed[0]; 
  
    console.log("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA : "+ user.data);

  // JSON.stringify() takes a JavaScript object and transforms it into a JSON string / SON.parse() takes a JSON string and transforms it into a JavaScript object
    return JSON.stringify("Best portfolio is :"+ user.index +'\n' +user.data)
  }

  if (result.intent.displayName === 'other') {
    //console.log(msg)
    //var res = await webservice.get_webServ_parameters(msg);
    

    //var obj = JSON.parse(res);
    //console.log(obj['test']);
    //return "A "+ "'"+obj['test']+"' ."+"\ A hello";
    return 'Helooooo'
  }


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
