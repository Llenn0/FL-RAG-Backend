import { useEffect, useState, type ChangeEvent } from "react";
import axios from "axios";

type UploadState = "idle" | "uploading" | "success" | "failed";
const BACKEND_URL = "http://127.0.0.1:5000/";

export default function FileUpload() {
  const [newFiles, setNewFiles] = useState<FileList | null>(null); // Files for upload that have yet to be uploaded
  const [uploadState, setUploadState] = useState<UploadState>("idle"); // Current status of the upload, defined above
  const [files, setFiles] = useState<String[]>([]); // Filenames for successfully uploaded files
  const [loading, setLoading] = useState<boolean>(true); // Status of fetching files
  const [error, setError] = useState<String | null>(null); // Error status of fetching files

  // Handles correctly displaying the file list
  const fetchFiles = async () => {
    try {
      setLoading(true);
      const response = await axios.get(BACKEND_URL + "files"); // Call backend file endpoint to read AWS
      setFiles(response.data.files);
      setError(null);
    } catch (err) {
      setError("Failed to fetch files");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchFiles();
  }, []);

  // Update new file list when files are selected
  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    if (event.target.files) {
      setNewFiles(event.target.files);
    }
  }

  // Upload selected files to backend server
  async function handleFileUpload() {
    if (!newFiles) return;

    // Create form data
    const formData = new FormData();
    for (let i = 0; i < newFiles.length; i++) {
      formData.append("files[]", newFiles[i]);
    }
    setUploadState("uploading");

    // Upload filse to backend with post request
    await axios
      .post(BACKEND_URL + "upload", formData, {})
      .then((res) => {
        console.log("Got response ", res.status);
        setUploadState("success");
      })
      .catch((err) => {
        console.log(err);
        setUploadState("failed");
      });
    setNewFiles(null); // Reset new files state for next upload
    await fetchFiles(); // Refresh the file list
  }

  return (
    <section className="Uploader">
      <div>
        <h2>Uploaded Files:</h2>
        {!loading && !error && (
          <ul>
            {files.map((file, index) => (
              <li key={index}>{file}</li>
            ))}
          </ul>
        )}
        {loading && <p>Loading files...</p>}
        {error && <p>{error}</p>}
      </div>
      <br></br>
      <h2>Upload files</h2>
      <input
        type="file"
        multiple
        accept=".pdf"
        onChange={handleFileChange}
      ></input>

      {newFiles && uploadState !== "uploading" && (
        <button onClick={handleFileUpload}>Upload</button>
      )}
      <div>
        {uploadState === "uploading" && <p>Uploading...</p>}
        {uploadState === "success" && <p>Success!</p>}
        {uploadState === "failed" && <p>Failed!</p>}
      </div>
    </section>
  );
}
