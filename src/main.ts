import './style.css';
import { Scanner } from './scanner';

const welcome = document.getElementById('welcome')!;
const scanner = new Scanner();

document.getElementById('choose-scan')!.addEventListener('click', () => {
  welcome.hidden = true;
  scanner.start();
});

document.getElementById('choose-scan')!.focus();
