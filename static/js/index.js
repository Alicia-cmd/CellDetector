function showdetection(){
    $("#detection_container").css("display","inherit");
    $("#detection_container").addClass("animated slideInLeft");
    setTimeout(function(){
        $("#detection_container").removeClass("animated slideInLeft");
    },800);
}
function closedetection(){
    $("#detection_container").addClass("animated slideOutLeft");
    setTimeout(function(){
        $("#detection_container").removeClass("animated slideOutLeft");
        $("#detection_container").css("display","none");
    },800);
}
function showmore(){
    $("#more_container").css("display","inherit");
    $("#more_container").addClass("animated slideInRight");
    setTimeout(function(){
        $("#more_container").removeClass("animated slideInRight");
    },800);
}
function closemore(){
    $("#more_container").addClass("animated slideOutRight");
    setTimeout(function(){
        $("#more_container").removeClass("animated slideOutRight");
        $("#more_container").css("display","none");
    },800);
}
function showcontact(){
    $("#contact_container").css("display","inherit");
    $("#contact_container").addClass("animated slideInUp");
    setTimeout(function(){
        $("#contact_container").removeClass("animated slideInUp");
    },800);
}
function closecontact(){
    $("#contact_container").addClass("animated slideOutDown");
    setTimeout(function(){
        $("#contact_container").removeClass("animated slideOutDown");
        $("#contact_container").css("display","none");
    },800);
}
setTimeout(function(){
    $("#loading").addClass("animated fadeOut");
    setTimeout(function(){
      $("#loading").removeClass("animated fadeOut");
      $("#loading").css("display","none");
      $("#box").css("display","none");
      $("#detection").removeClass("animated fadeIn");
      $("#contact").removeClass("animated fadeIn");
      $("#more").removeClass("animated fadeIn");
    },1000);
},1500);
