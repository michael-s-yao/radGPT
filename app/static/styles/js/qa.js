function NonNull(item) {
  if (item === null || item === undefined)
    throw new Error();
  return item;
}

let studies = document.getElementById("imaging_studies");

function checkIfFinished() {
  const options = [...studies.options].map((opt) => opt.value);
  let isFinished = [...document.querySelectorAll(".question")]
    .every((x) => options.includes(x.querySelector("input").value));
  if (isFinished)
    document.getElementById("submit").style.display = "block";
  else
    document.getElementById("submit").style.display = "none";
}

function linkDatalist(question) {
  let input = NonNull(question.querySelector("input"));
  input.onfocus = function () {
    studies.style.display = "block";
    input.style.borderRadius = "8px 8px 0 0";
  };

  for (let option of studies.options) {
    option.onclick = function () {
      input.value = option.value;
      studies.style.display = "none";
      input.style.borderRadius = "8px";
    }
  };

  input.oninput = function() {
    currentFocus = -1;
    var text = input.value.toUpperCase();
    for (let option of studies.options) {
      if (option.value.toUpperCase().indexOf(text) > -1)
        option.style.display = "block";
      else
        option.style.display = "none";
    };
  };

  input.onkeydown = function(e) {
    input.style.borderColor = null;
    if (e.keyCode == 40) {
      currentFocus++;
      addActive(studies.options);
    } else if (e.keyCode == 38) {
      currentFocus--;
      addActive(studies.options);
    } else if (e.keyCode == 13) {
      e.preventDefault();
    if (currentFocus > -1 && studies.options)
      studies.options[currentFocus].click();
    }
  }

  input.onkeyup = checkIfFinished;
};

function unlinkDatalist() {
  [...document.querySelectorAll(".question")]
    .map(function (input) {
      input.onfocus = null;
      [...studies.options].map(function (option) { option.onclick = null; });
      input.oninput = null;
      input.onkeydown = null;
      input.onkeyup = null;
    });
};

function addActive(x) {
  if (!x)
    return false;
  removeActive(x);
  if (currentFocus >= x.length)
    currentFocus = 0;
  if (currentFocus < 0)
    currentFocus = (x.length - 1);
  x[currentFocus].classList.add("active");
}

function removeActive(x) {
  for (var i = 0; i < x.length; i++)
    x[i].classList.remove("active");
}

function validateInput(event) {
  const options = [...studies.options]
    .filter((x) => x.value.length > 0)
    .map((x) => x.value.substring(0, x.value.length));
  const activeElement = NonNull(document.querySelector(".question.active"));
  const input = NonNull(activeElement.querySelector("input"));
  const isValid = options.includes(input.value);
  if (isValid) {
    if (event.target.id.toLowerCase() === "next")
      activeView = Math.min(activeView + 1, numQuestions);
    else
      activeView = Math.max(activeView - 1, 1);
    setView(activeView);
  } else {
    input.style.borderColor = "var(--red)"; 
  }
}

function resetView() {
  [...document.querySelectorAll(".question")]
    .map((x) => x.classList.remove("active"));
  [...document.querySelectorAll("button.nav")]
    .map((x) => x.style.display = null);
  unlinkDatalist();
}

function setView(idx) {
  resetView();
  let view = [...document.querySelectorAll(".question")]
    .find((x) => x.id === "Q" + idx.toString() + "-box");
  NonNull(view).classList.add("active");
  linkDatalist(NonNull(view));
  if (idx === 1)
    document.querySelector("button#back").style.display = "none";
  if (idx === numQuestions)
    document.querySelector("button#next").style.display = "none";
}

function submit() {
  if (!confirm("I confirm that I have completed this task to the best of my ability."))
    return;
  resetView();
  const answers = [...document.querySelectorAll(".question")]
    .map(function (x) {
      y = new Object();
      y.question = NonNull(parseInt(x.id.replace(/[^0-9]/g, "")));
      y.answer = NonNull(x.querySelector("input")).value;
      return y;
    });
  document.getElementById("submit").style.display = "none";
  document.getElementById("success").style.display = "block";
  fetch("/submit", {
    method: "POST",
    body: JSON.stringify(answers),
    headers: {"Content-type": "application/json; charset=UTF-8"}
  });
}

var activeView = 1;
var currentFocus = -1;
const numQuestions = document.querySelectorAll(".question").length;
setView(activeView);
[...document.querySelectorAll("button.nav")]
  .map((x) => x.addEventListener("click", validateInput));
document.addEventListener("click", checkIfFinished);
document.getElementById("submit").onclick = submit;
