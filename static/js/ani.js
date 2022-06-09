$(document).ready(function(){
$('tr').hide();
$('tr').each(function(index){
	$(this).delay(index*500).show(500);
	$('tr').each(function(index){
	$(this).delay(index*500).hide(500);
	
});
	
});

});
