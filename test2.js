/**
 * @typedef {object} Usuario
 * @property {number} id - El ID único del usuario.
 * @property {string} nombre - El nombre completo del usuario.
 * @property {string} email - La dirección de correo electrónico del usuario.
 * @property {boolean} estaActivo - Indica si el usuario está activo.
 */

/**
 * Lee un archivo de texto de forma asíncrona y devuelve su contenido.
 * @param {string} rutaArchivo - La ruta completa o relativa al archivo.
 * @param {string} [codificacion='utf8'] - La codificación del archivo (opcional, por defecto 'utf8').
 * @returns {Promise<string>}
 */
function leerArchivoAsincrono(rutaArchivo, codificacion = "utf8") {
  const fs = require("fs");

  return new Promise((resolve, reject) => {
    fs.readFile(rutaArchivo, codificacion, (err, data) => {
      if (err) {
        reject(err);
      } else {
        resolve(data);
      }
    });
  });
}

/**
 * Procesa la información de un usuario.
 * @param {Usuario} usuario - El objeto de usuario a procesar.
 * @returns {string} Un mensaje de bienvenida para el usuario.
 */
function saludarUsuario(usuario) {
  return `¡Hola, ${usuario.nombre}! Tu email es ${usuario.email}.`;
}

// --- Uso de las funciones ---

const nombreArchivo = "./julia-titanic-notebook/input/test.csv";

// Ejemplo con JSDoc en el callback
/**
 * Callback para manejar el resultado de la lectura del archivo.
 * @callback ManejarLecturaCallback
 * @param {Error|null} err - El objeto de error si hubo un problema, o null si fue exitoso.
 * @param {string|null} data - El contenido del archivo si fue exitoso, o null si hubo un error.
 */

const fs = require("fs");

/** @type {ManejarLecturaCallback} */
const miCallbackLectura = (err, data) => {
  if (err) {
    console.error("Error al leer el archivo:", err);
    return;
  }
  console.log("Contenido del archivo (usando JSDoc en el callback):");
  console.log(data);
};

fs.readFile(nombreArchivo, "utf8", miCallbackLectura);

// Ejemplo usando la función con promesa (mejor para asincronía)
leerArchivoAsincrono(nombreArchivo)
  .then((contenido) => {
    console.log("\nContenido del archivo (con promesa y JSDoc):");
    console.log(contenido);
  })
  .catch((error) => {
    console.error("\nError al leer el archivo (con promesa y JSDoc):", error);
  });

// Ejemplo de uso de la función saludarUsuario
/** @type {Usuario} */
const miUsuario = {
  id: 1,
  nombre: "Ana Pérez",
  email: "ana.perez@example.com",
  estaActivo: true,
};

console.log(saludarUsuario(miUsuario));

// JSDoc para variables
/** @type {number} */
let contador = 0;

/** @type {string[]} */
const nombres = ["Juan", "María", "Pedro"];

/** @type {Array<Object.<string, any>>} */
const listaMixta = [{ a: 1 }, { b: "dos" }]; // Array de objetos con propiedades de cualquier tipo
