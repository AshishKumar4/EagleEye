

function createFlagEntry(data)
{
    var dd = JSON.parse(data);
    var s = "";
    for(var i = 0; i < dd['data'].length; i++)
    {
        s += '<option value="' + dd['data'][i] + '">' + dd['data'][i] + '</option>'.replace(/undefined/g, "");
    }
    return s;
}

function loadUserKeys(obj)
{
    var dd = document.getElementById("api-keys");
    dd.innerHTML += createFlagEntry(obj);
}   

function inferText(user)
{
    var text = document.getElementById("test-text").value;
    var key = document.getElementById("test-key").value;
    var postdata = {text:text, apikey:key, user:user};
    var xhttp = new XMLHttpRequest();
    xhttp.onreadystatechange = function () 
    {
        if (this.readyState == 4 && this.status == 200) 
        {
            //document.getElementById("hint-box").innerHTML = this.responseText;
            alert(this.responseText);
            var dd = JSON.parse(this.responseText);
            var inf = document.getElementById("inference");
            inf.innerHTML = dd[0];
            inf.style.backgroundColor = dd[1];
            var callid = document.getElementById("callid");
            callid.innerHTML = dd[2];
            document.getElementById("feedback").style.visibility = "visible";
        }
    };
    xhttp.open("POST", "/inferAPI/universal", true);
    xhttp.send(JSON.stringify(postdata));
}

function feedback(user, feedback)
{
    var key = document.getElementById("test-key").value;
    var callid = document.getElementById("callid").innerHTML;
    var postdata = {apikey:key, user:user, feedback:feedback, callid:callid};
    var xhttp = new XMLHttpRequest();
    xhttp.onreadystatechange = function () 
    {
        if (this.readyState == 4 && this.status == 200) 
        {
            //document.getElementById("hint-box").innerHTML = this.responseText;
            alert(this.responseText);
            document.getElementById("feedback").style.visibility = "hidden";
            document.getElementById("saveModel").style.visibility = "visible";
        }
    };
    xhttp.open("POST", "/inferAPI/feedback", true);
    xhttp.send(JSON.stringify(postdata));
}

function saveModel(user)
{
    var key = document.getElementById("test-key").value;
    var postdata = {apikey:key, user:user};
    var xhttp = new XMLHttpRequest();
    xhttp.onreadystatechange = function () 
    {
        if (this.readyState == 4 && this.status == 200) 
        {
            //document.getElementById("hint-box").innerHTML = this.responseText;
            alert(this.responseText);
            document.getElementById("saveModel").style.visibility = "hidden";
        }
    };
    xhttp.open("POST", "/inferAPI/save", true);
    xhttp.send(JSON.stringify(postdata));
}

function loadModel(user)
{
    var key = document.getElementById("test-key").value;
    var postdata = {apikey:key, user:user};
    var xhttp = new XMLHttpRequest();
    xhttp.onreadystatechange = function () 
    {
        if (this.readyState == 4 && this.status == 200) 
        {
            //document.getElementById("hint-box").innerHTML = this.responseText;
            alert(this.responseText);
            document.getElementById("saveModel").style.visibility = "hidden";
        }
    };
    xhttp.open("POST", "/inferAPI/save", true);
    xhttp.send(JSON.stringify(postdata));
}

function createModel(user)
{
    var dat = document.getElementById("test-key").value.split(':');
    var templatekey = dat[0];
    var newkey = dat[1];
    var postdata = {templatekey:templatekey, newuser:user, originaluser:user, newkey:newkey};
    var xhttp = new XMLHttpRequest();
    xhttp.onreadystatechange = function () 
    {
        if (this.readyState == 4 && this.status == 200) 
        {
            //document.getElementById("hint-box").innerHTML = this.responseText;
            alert(this.responseText);
            //document.getElementById("saveModel").style.visibility = "hidden";
        }
    };
    xhttp.open("POST", "/inferAPI/create", true);
    xhttp.send(JSON.stringify(postdata));
}