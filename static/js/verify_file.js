var imageInput = document.getElementById('image');

imageInput.addEventListener('change', function () {
  var file = imageInput.files[0]; // Récupère le fichier sélectionné

  // Vérifie que le fichier existe et a l'extension TIF
  if (file && file.type === 'image/tiff') {
    // Le fichier est valide
  } else {
    // Le fichier n'est pas valide
    alert('Seuls les fichiers TIF sont autorisés');
    imageInput.value = ''; // Efface la valeur du champ d'entrée pour éviter une soumission incorrecte
  }
});

function cliquer_lien() {
  document.getElementById("lien").innerHTML = '<a href="resultat.csv" download>Télécharger CSV</a>';
}
