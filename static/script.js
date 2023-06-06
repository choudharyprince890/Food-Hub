console.log("js conected")
const desp = document.querySelector(".desp").innerHTML;
const ingr = document.querySelector(".ingr").innerHTML;
const inst = document.querySelector(".inst").innerHTML;
// document.getElementById("showRecommendations").style.display = "none"
// console.log(desp,"de",ingr,"ing",inst);
if (desp.length > 180) {
  document.querySelector(".readmore").style.display = "inline-block";
  let arr = desp.slice(0, 180);
  document.querySelector(".desp").textContent = arr;
}

if (inst.length > 180) {
  document.querySelector(".readmore1").style.display = "inline-block";
  let arr1 = inst.slice(0, 180);
  document.querySelector(".inst").textContent = arr1;
}

if (ingr.length > 180) {
  document.querySelector(".readmore2").style.display = "inline-block";
  let arr2 = ingr.slice(0, 180);
  document.querySelector(".ingr").textContent = arr2;
}

document.querySelector(".readmore").onclick = function () {
  let arr12 = desp.slice(0, desp.length - 1);
  document.querySelector(".desp").textContent = arr12;
  document.querySelector(".readmore").style.display = "none";
};

document.querySelector(".readmore1").onclick = function () {
  let arr13 = inst.slice(0, inst.length - 1);
  document.querySelector(".inst").textContent = arr13;
  document.querySelector(".readmore1").style.display = "none";
};

document.querySelector(".readmore2").onclick = function () {
  let arr14 = ingr.slice(0, ingr.length - 1);
  document.querySelector(".ingr").textContent = arr14;
  document.querySelector(".readmore2").style.display = "none";
};


//form-submit-btn
document.getElementById("formSubmit").onclick = function(e){
  console.log("botton pressed")
  e.preventDefault()
  document.getElementById('showRecommendaion').style.display = "inline-block";

}









