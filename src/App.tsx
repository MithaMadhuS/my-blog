// import React from 'react';
// // import logo from './logo.svg';
// import './App.css';

// function App() {
//   return (
//     // <div className="App">
//     //   <header className="App-header">
//     //     <img src={logo} className="App-logo" alt="logo" />
//     //     <p>
//     //       Edit <code>src/App.tsx</code> and save to reload.
//     //     </p>
//     //     <a
//     //       className="App-link"
//     //       href="https://reactjs.org"
//     //       target="_blank"
//     //       rel="noopener noreferrer"
//     //     >
//     //       Learn React
//     //     </a>
//     //   </header>
//     // </div>
//     <div>
//       HEllo
//     </div>
//   );
// }

// export default App;


import React from 'react';
import BlogPost from './BlogPost';
import RAG_writeup from './posts/RAG_writeup.md';

function App() {
  return (
    <div className="App">
      <BlogPost content={RAG_writeup} title="Unlocking Advanced RAG: Where the Real Magic Begins" date="2025-05-15"/>
    </div>
  );
}

export default App;
