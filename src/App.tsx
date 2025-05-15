import Chatbot from "./components/Chatbot";
import FileUpload from "./components/FileUpload";

function App() {
  return (
    <>
      <h1>FinLegal Document RAG</h1>
      <section className="container">
        <div>
          <FileUpload />
        </div>
        <Chatbot />
      </section>
    </>
  );
}

export default App;
