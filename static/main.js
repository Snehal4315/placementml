// console.log('hii');
var x;
document.getElementById('error').style.color = 'red';
document.getElementById('error').style.fontSize = '30px'

x = document.getElementById('pre1').innerHTML;
if(x == 'Placed'){
    document.getElementById('pre1').style.color = 'green';
}
else{
    document.getElementById('pre1').style.color = 'red'
}
